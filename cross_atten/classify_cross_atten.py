import argparse
import os
import sys
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
import torch
from torch import nn
from tqdm import tqdm
import sys; sys.path.append('./')
from pytorch3dunet.unet3d.model import Residual_mid_UNet3D
from classify.classifier import Combine_classfier_cross
from utils.common import copy_yaml_to_folder_auto, get_parameters, load_config
from dataloader.pic_table_loader import classi_dataloader
from os.path import join as j
from torch.nn import functional as F
from corss_ft_transformer import FTTransformer_cross
from torchmetrics import Recall, F1Score

def main(args):
    cf = load_config(args.config_path)
    is_debug = True
    if not is_debug:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
    kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    train_dataloader = classi_dataloader(cf['train_path'], 
                                        cf['img_sz'], 
                                        cf['train_bc'], 
                                        cf['table_path'], 
                                        True)
    val_dataloader = classi_dataloader(cf['eval_path'], 
                                        cf['img_sz'], 
                                        cf['eval_bc'], 
                                        cf['table_path'], 
                                        True)
    accelerator = Accelerator(**get_parameters(Accelerator, cf), kwargs_handlers=[kwargs])
    ref_model = Residual_mid_UNet3D(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    ref_model.load_state_dict(torch.load('/home/chenyifei/data/CYFData/FZJ/weights_of_models/weights_ADNI/weights/exp_2_10/model_save/model.pt'))

    table_df = train_dataloader.dataset.table_df
    ft_model = FTTransformer_cross(
        categories = table_df['num_cat'],      # tuple containing the number of unique values within each category
        num_continuous = table_df['num_cont'],                # number of continuous values
        dim = cf['dim'],                           # dimension, paper set at 32
        dim_out = cf['dim_out'],                        # binary prediction, but could be anything
        depth = cf['depth'],                          # depth, paper recommended 6
        heads = cf['heads'],                          # heads, paper recommends 8
        attn_dropout = cf['attn_dropout'],                 # post-attention dropout
        ff_dropout = cf['ff_dropout'], 
        dim_head = cf['dim'] // cf['heads'] , 
        dim_cross = cf['dim_cross']
        )
    ref_model.eval()
    ref_model = ref_model.to(accelerator.device)

    recall_metric = Recall(average='macro', task='binary').to(accelerator.device)
    f1_metric = F1Score(average='macro',task='binary').to(accelerator.device)
    model = Combine_classfier_cross(16)
    model_params = list(model.parameters())
    ft_model_params = list(ft_model.parameters())

    # 将两个模型的参数连接成一个单一的张量列表
    all_params = model_params + ft_model_params

    # 创建优化器并传递连接后的参数列表
    optimizer = torch.optim.Adam(all_params, lr=1e-4)
    val_interval = cf['val_inter']
    save_interval = cf['save_inter']
    loss_fn = nn.BCELoss()

    model, ft_model,  optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, ft_model, optimizer, train_dataloader, val_dataloader
    )
    global_step = 0


    log_dir = j(dir, 'train_loss.txt') if not is_debug else 'debug.txt'
    file = open(log_dir, "w")
    for epoch in range(cf['num_epochs']):
        model.train()
        ft_model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        for step, batch in enumerate(train_dataloader):
            x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label']
    
            if x.shape[1] != 1:
                print("Wrong! Got the first channel!", batch['name'])
                x = x[:,:1,...]
            with accelerator.accumulate(model):
                with torch.no_grad():
                    list1, list2, _ = ref_model(x, output_mid=True)
                img_feature = model(list1, list2)
                pred = ft_model(x_cat, x_num, condition=img_feature)
                # print("Pred: ", pred, "y: ", y)
                cla_loss = loss_fn(F.sigmoid(pred.squeeze(1)), y.float())
                accelerator.backward(cla_loss)
                # for param in all_params:
                #     torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
                for name, param in model.named_parameters():
                    # 如果参数没有梯度，打印相关信息
                    if param.grad is None:
                        print(f"Parameter {name} has no gradient.")
                for name, param in ft_model.named_parameters():
                    # 如果参数没有梯度，打印相关信息
                    if param.grad is None:
                        print(f"Parameter {name} has no gradient.")
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"cla_loss": cla_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1 

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            val_model = accelerator.unwrap_model(model).eval()
            val_ft_model = accelerator.unwrap_model(ft_model).eval()
            
            if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
                correct = 0
                losses = 0
                total = 0
                for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    x, x_cat, x_num, y = batch['image'], batch['cate_x'], batch['conti_x'], batch['label'].float()
                    with torch.no_grad():
                        list1, list2, _ = ref_model(x, output_mid=True)
                        img_feature = val_model(list1, list2)
                        pred = val_ft_model(x_cat, x_num, condition=img_feature)
                    pred = F.sigmoid(pred)
                    loss = loss_fn(pred.squeeze(1), y)
                    pred_labels = pred.round()
                    total += y.size(0)
                    correct += (pred_labels.squeeze(1) == y).sum().item()
                    losses += loss.item()
                    recall_metric(pred_labels.squeeze(1), y)
                    f1_metric(pred_labels.squeeze(1), y)
                accuracy=(100 * correct / total)
                recall=recall_metric.compute().item()
                f1=f1_metric.compute().item()
                recall_metric.reset()
                f1_metric.reset()
                validation_loss=(losses / total)
                # print(f'Acc: {accuracy: .4f}', file=file) 
                # print(f'Recall: {recall: .4f}', file=file)
                # print(f'F1: {f1:.4f}', file=file)
                # print(f'Val_loss: {validation_loss: .4f} \n', file=file) 
                logs = dict(accuracy=accuracy, recall=recall, f1=f1, validation_los=validation_loss)
                file.write(f"Epoch {epoch+1}: Acc: {accuracy: .4f}% Recall: {recall: .4f} F1: {f1:.4f} Val_loss: {validation_loss: .4f} \n" )
                file.flush()
                accelerator.log(logs, step=epoch+1)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/trans_classi_cross_config.yaml')

    args = parser.parse_args()
    main(args)