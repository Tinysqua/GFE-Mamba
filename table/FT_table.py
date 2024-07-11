
import re
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import sys;sys.path.append('./')
from monai.networks.nets.vit import ViT
from tqdm import tqdm
from tab_transformer_pytorch import FTTransformer
from torch import nn
class RegressionColumnarDataset(data.Dataset):
    def __init__(self, df, cats, y):
        
        
        self.dfcats = df[cats] 
        self.dfconts = df.drop(cats, axis=1) 
        
        
        self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(np.int64) #tpye: numpy.ndarray
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32) #tpye: numpy.ndarray
        self.y = y.astype(np.int64)
        
        
    def __len__(self): return len(self.y)
 
    def __getitem__(self, idx):
        
        return [self.cats[idx], self.conts[idx], self.y[idx]]
    
def has_letters(string):
    if not isinstance(string, str):
        return False
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    if match:
        return True
    else:
        return False
    
def discovery_mix(df):
    str_columns = df.select_dtypes(include='object').columns
    # 检查包含字符串类型的列是否具有混合类型
    mixed_columns = []
    for column in str_columns:
        # if df[column].apply(lambda x: isinstance(x, str)).sum() < len(df[column]):
        if df[column].apply(has_letters).sum() > 0:
            mixed_columns.append(column)
    # 打印含有混合类型的列
    print("混合类型的列：", mixed_columns)
    return mixed_columns

def prepare_table(table_path):
    mri_df = pd.read_csv(table_path)
    drop_list = ['RID', 'D1', 'D2', 'SITE', 'DX',
                 'VERSION_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'FLDSTRENG_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16']
    info_list = ['DXCHANGE']
    for i, col in enumerate(mri_df.columns):
        if 'DATE' in col or 'stamp' in col:
            drop_list.append(col)
        # if mri_df[col].isna().mean() > 0.5:
        #     print(f'drop the column {col} with too many blanks')
        #     drop_list.append(col)
    table_info = LabelEncoder().fit_transform(mri_df[info_list])
    # table_info = mri_df[info_list]
    output_dim = table_info.max() + 1
    mri_df = mri_df.drop(drop_list + info_list, axis=1)
    mixed_columns = discovery_mix(mri_df)
    num_columns = [x for x in mri_df.columns if x not in mixed_columns]
    mri_df[mixed_columns] = mri_df[mixed_columns].fillna('NA').astype('category')
    num_cat = []
    for col in mri_df.columns:
        if col in mixed_columns:
            mri_df[col] = LabelEncoder().fit_transform(mri_df[col])
            num_cat.append(len(mri_df[col].unique()))
        else:
            mri_df[col] = pd.to_numeric(mri_df[col], errors='coerce')
            mri_df[col] = mri_df[col].fillna(0)
    sc = StandardScaler()
    sc.fit(mri_df[num_columns])
    mri_df[num_columns] = sc.transform(mri_df[num_columns])
    # for col in mri_df.columns:
    #     if col in mixed_columns:
    #         mri_df[col] = mri_df[col].fillna('NA')
    #         mri_df[col] = mri_df[col].astype('category')
    #         # mri_df[col] = LabelEncoder().fit_transform(mri_df[col])
    #     else:
    #         mri_df[col] = pd.to_numeric(mri_df[col], errors='coerce')
    #         mri_df[col] = mri_df[col].fillna(0)
            
    X_train, X_val, y_train, y_val = train_test_split(mri_df, table_info, test_size=0.20, random_state=42,shuffle=True )
    train_dt = RegressionColumnarDataset(X_train, mixed_columns, y_train)
    val_dt = RegressionColumnarDataset(X_val, mixed_columns, y_val)
    params = {'batch_size': 8,
            'shuffle': True, 
            'drop_last': True}
    traindl = data.DataLoader(train_dt, **params) 
    valdl = data.DataLoader(val_dt, **params) 

    device = 'cuda'
    criterion =nn.CrossEntropyLoss()
    model = FTTransformer(
    categories = num_cat,      # tuple containing the number of unique values within each category
    num_continuous = len(num_columns),                # number of continuous values
    dim = 512,                           # dimension, paper set at 32
    dim_out = output_dim,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1  
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    TOTAL_EPOCHS=1000
    losses = []
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        for i, (x_cat, x_cont, y) in tqdm(enumerate(traindl), total=len(traindl)):
            x_cat = x_cat.to(device) 
            x_cont = x_cont.to(device) #输入必须未float类型
            y = y.long().to(device) #结果标签必须未long类型
            opt.zero_grad()
            outputs = model(x_cat, x_cont)
            #计算损失函数
            loss = criterion(outputs, y.squeeze())
            loss.backward()
            opt.step()
            losses.append(loss.cpu().data.item())
        print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, np.mean(losses)))
        if (epoch+1) % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            for i,(x_cat, x_cont, y) in enumerate(valdl):
                x_cat = x_cat.to(device) 
                x_cont = x_cont.to(device) #输入必须未float类型
                y = y.long().to(device).squeeze() #结果标签必须未long类型
                with torch.no_grad():
                    outputs = model(x_cat, x_cont)
                _, predicted = torch.max(outputs.data, 1)
                total += x_cat.shape[0]
                correct += (predicted == y).sum()
            print('准确率: %.4f %%' % (100 * correct / total))
    

# prepare_table('/home/chenyifei/data/CYFData/FZJ/ADNI_MRI2AD&neg/TADPOLE_D1_D2.csv')
prepare_table('/home/chenyifei/data/CYFData/FZJ/ADNI_MRI2AD&neg/filt.csv')