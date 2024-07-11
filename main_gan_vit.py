import sys; sys.path.append('./')
import argparse
from accelerate import Accelerator
from tqdm import tqdm
from pytorch3dunet.unet3d.model import Residual_mid_UNet3D_vit
from generative.networks.nets import PatchDiscriminator
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from utils.common import load_config, copy_yaml_to_folder_auto, get_parameters, plt_mri_pet, see_mri_pet
from dataloader.threed_loader import form_dataloader
import torch
from accelerate.utils import DistributedDataParallelKwargs
from torch.nn import functional as F
from os.path import join as j
import os
from torchvision.utils import save_image

def main(args):
    cf = load_config(args.config_path)
    if not cf['is_debug']:
        dir = copy_yaml_to_folder_auto(args.config_path, cf['project_dir'])
        cf['project_dir'] = dir
    kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    train_dataloader = form_dataloader(cf['train_path'], 
                                        cf['img_sz'], 
                                        cf['train_bc'], 
                                        True)
    val_dataloader = form_dataloader(cf['eval_path'], 
                                        cf['img_sz'], 
                                        cf['eval_bc'], 
                                        False)
    accelerator = Accelerator(**get_parameters(Accelerator, cf), kwargs_handlers=[kwargs])
    model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    discriminator = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    kernel_size=4,
    padding=1,)
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=0.25).to(accelerator.device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001
    optimizer_g = torch.optim.Adam(model.parameters(), 1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

    val_interval = cf['val_inter']
    save_interval = cf['save_inter']
    autoencoder_warm_up_n_epochs = 10
    
    if len(cf['log_with']):
        accelerator.init_trackers('train_example')
    model, discriminator, optimizer_g, optimizer_d, train_dataloader, val_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader, val_dataloader
    )
    global_step = 0
    for epoch in range(cf['num_epochs']):

            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch+1}")
            for step, batch in enumerate(train_dataloader):
                condition, target = batch['image'], batch['label']
                if condition.shape[1] != 1:
                    print("Wrong! Got the first channel!", batch['name'])
                    condition = condition[:,:1,...]
                    
                with accelerator.accumulate(model):
                    reconstruction = model(condition)
                    recons_loss = F.l1_loss(reconstruction.float(), target.float())
                    p_loss = perceptual_loss(reconstruction.float(), target.float())
                    loss_g = recons_loss + (perceptual_weight * p_loss)
                    
                    if epoch+1 > autoencoder_warm_up_n_epochs:
                        logits_fake = discriminator(reconstruction.contiguous())[-1]
                        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                        loss_g += adv_weight * generator_loss
                    
                    accelerator.backward(loss_g)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_g.step()
                    optimizer_g.zero_grad()
                    
                if epoch+1 > autoencoder_warm_up_n_epochs:
                    with accelerator.accumulate(discriminator):
                        with torch.no_grad():
                            recon = model(condition)
                        logits_fake = discriminator(recon.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(target.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                        
                        loss_d = adv_weight * discriminator_loss
                        accelerator.backward(loss_d)
                        optimizer_d.step()
                        optimizer_d.zero_grad()
            
                progress_bar.update(1)
                logs = {"g_loss": loss_g.detach().item()}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1 

                
            if accelerator.is_main_process:
                val_model = accelerator.unwrap_model(model)
                if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
                    for i, batch in enumerate(val_dataloader):
                        with torch.no_grad():
                            condition, target = batch['image'], batch['label']
                            val_recon = val_model(condition)
                        images = torch.cat([condition, target, val_recon], dim=-2)
                        save_pic_dir = j(cf['project_dir'], f'results_save/{epoch+1}')
                        os.makedirs(save_pic_dir, exist_ok=True)
                        save_image(see_mri_pet(images), j(save_pic_dir, f'{i+1}.png'))
                        # with torch.no_grad():
                        #     batch = next(iter(val_dataloader))
                        #     val_recon = val_model(batch["image"])
                        #     image = batch['image'][:, 0, ...]
                        #     label = batch["label"][:, 0, ...]
                        #     val_recon = val_recon[:, 0, ...]
                        #     val_recon = torch.cat([image, label, val_recon], dim=-2)
                        #     for i in range(len(val_recon)):
                        #         tensor = val_recon[i]
                        #         save_pic_dir = j(cf['project_dir'], f'results_save/{epoch+1}/{i}.png')
                        #         os.makedirs(os.path.dirname(save_pic_dir), exist_ok=True)
                        #         plt_mri_pet(tensor.cpu(), save_pic_dir)

                    
                if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
                    model_save_dir = j(dir, 'model_save/model.pt')
                    os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
                    torch.save(val_model.state_dict(), model_save_dir)



            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/main_gan_vit_config.yaml')

    args = parser.parse_args()
    main(args)