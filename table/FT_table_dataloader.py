import sys;sys.path.append('./')
import numpy as np

import torch
from tqdm import tqdm

from tab_transformer_pytorch.ft_transformer import FTTransformer

from dataloader.pic_table_loader import classi_dataloader
from torch import nn
from torch.nn import functional as F
from torchmetrics import Recall, F1Score

train_dataloader = classi_dataloader('/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_3year_train', 
                                         (160, 160, 96), batch_size=4, 
                                         table_path='/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ct_2&5_3year.csv')
val_dataloader = classi_dataloader('/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_3year_test', 
                                         (160, 160, 96), batch_size=8, 
                                         table_path='/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ct_2&5_3year.csv')
table_df = train_dataloader.dataset.table_df
device = 'cuda'
criterion = nn.BCELoss()
model = FTTransformer(
    categories = table_df['num_cat'],      # tuple containing the number of unique values within each category
    num_continuous = table_df['num_cont'],                # number of continuous values
    dim = 512,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1, 
    dim_head=512 // 8  
    ).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
TOTAL_EPOCHS=200
cla_losses = []
file =  open("train_log_no_dup.txt", "w")

recall_metric = Recall(average='macro', task='binary').to(device)
f1_metric = F1Score(average='macro',task='binary').to(device)
for epoch in range(TOTAL_EPOCHS):
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x_cat = batch['cate_x'].to(device) 
        x_cont = batch['conti_x'].to(device) #输入必须未float类型
        y = batch['label'].float().to(device) #结果标签必须未long类型
        opt.zero_grad()
        outputs = model(x_cat, x_cont)
        #计算损失函数
        loss = criterion(F.sigmoid(outputs.squeeze()), y)
        loss.backward()
        opt.step()
        cla_losses.append(loss.cpu().data.item())
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, np.mean(cla_losses)))
    if (epoch+1) % 1 == 0:
        model.eval()
        correct = 0
        total = 0
        losses = 0
        for i,batch in enumerate(val_dataloader):
            x_cat = batch['cate_x'].to(device) 
            x_cont = batch['conti_x'].to(device) #输入必须未float类型
            y = batch['label'].float().to(device) #结果标签必须未long类型
            with torch.no_grad():
                outputs = model(x_cat, x_cont)
            predicted = torch.where(F.sigmoid(outputs) > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            # _, predicted = torch.max(outputs.data, 1)
            loss = criterion(F.sigmoid(outputs.squeeze()), y)
            losses += loss.item()
            total += x_cat.shape[0]
            correct += (predicted.squeeze() == y).sum()
            recall_metric(predicted.squeeze(), y)
            f1_metric(predicted.squeeze(), y)
        print('准确率: %.4f %%' % (100 * correct / total))
        logs = dict(accuracy=(100 * correct / total).item(), recall=recall_metric.compute().item(), f1=f1_metric.compute().item(), validation_los=(losses / total))
        file.write(f"第{epoch}轮训练: " + str(logs) + "\n")
        file.flush()
file.close()