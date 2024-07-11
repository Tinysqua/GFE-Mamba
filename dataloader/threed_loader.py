import sys; sys.path.append('./')
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils import data
import torch
import os
from os.path import join
from glob import glob
import re
from utils.data_normalization import adaptive_normal
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)
from table.deal_table import prepare_table
from utils.common import date_difference
import pandas as pd

def read_nii(ni_path, desired_shape=(160, 160, 96)):
    img = nib.load(ni_path)
    data = img.get_fdata()
    desired_depth = desired_shape[2]
    desired_width = desired_shape[1]
    desired_height = desired_shape[0]
    
    current_depth = data.shape[2]
    current_width = data.shape[1]
    current_height = data.shape[0]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    return zoom(data, (height_factor, width_factor, depth_factor), order=1)


class MRI2PET_dataset(data.Dataset):
    def __init__(self, data_path, desired_shape=(160, 160, 96)):
        super(MRI2PET_dataset, self).__init__()
        subject_path = os.listdir(data_path)
        self.parrent_path = data_path
        self.sub_path = subject_path
        self.desired_shape = desired_shape
        
        self.start_transformer = LoadImaged(keys=['image', 'label'])
        
        self.transformer = Compose(
            [
            EnsureChannelFirstd(keys=['image', 'label']),
            CropForegroundd(keys=['label'], source_key='label'),
            Resized(keys=['image', 'label'], spatial_size=desired_shape), 
            # ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=7000, b_min=-1.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=['label'], a_min=0.0, a_max=2, b_min=-1.0, b_max=1.0, clip=True),

            ToTensord(keys=['image', 'label'])
            ])
        
    def __getitem__(self, index, suffix='.nii.gz'):
        subject = join(self.parrent_path, self.sub_path[index])
        mri_path = join(subject, f'mri{suffix}')
        pet_path = join(subject, f'pet{suffix}')
        batch = self.start_transformer(dict(image=mri_path, label=pet_path))
        batch['image'] = adaptive_normal(batch['image'])
        # batch['label'] = adaptive_normal(batch['label'])
        batch = self.transformer(batch)
        batch['name'] =  mri_path
        # print(subject)
        return batch
        
    def __len__(self):
        return len(self.sub_path)

class MRI_classify(data.Dataset):
    def __init__(self, data_path, table_path='', 
                 desired_shape=(160, 160, 96)):
        super(MRI_classify, self).__init__()
        self.mri_nii = glob(join(data_path, '*.nii.gz'))
        self.start_transformer = LoadImaged(keys=['image'])
        
        self.transformer = Compose(
            [
            EnsureChannelFirstd(keys=['image']),
            Resized(keys=['image'], spatial_size=desired_shape), 
            ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1000, b_min=-1.0, b_max=1.0, clip=True),

            ToTensord(keys=['image'])
            ])
        self.import_table = len(table_path)
        if self.import_table:
            # self.table_dict = pd.read_csv(table_path)
            self.table_dict = prepare_table(table_path)
            for i, path in enumerate(self.mri_nii):
                if self.find_index(mri_path=path) == None:
                    self.mri_nii.pop(i)
            
        
    def find_row(self, ID, current_datetime):
        info = self.table_dict['info']
        min  = min_orig = 30
        min_index = -1
        difference = -1
        for index, row in info.iterrows():
            idInCsv, dateInCsv = row.iloc[0], row.iloc[1]
            if ID != idInCsv:
                continue
            difference = date_difference(dateInCsv, current_datetime)
            if min > difference:
                min = difference
                print(min)
                min_index = index

            if min <= 30 :
                break
        if min < min_orig :
            print("找到日期误差小于30天对应数据！")
            return min_index
        else:
            print("找不到日期误差小于30天"+" 对应数据(匹配数据信息：ID:"+str(ID)+"||date："+str(current_datetime)+")！")
            print(f"最小的日期差距是： {difference}")
            return (ID, current_datetime)

    def __getitem__(self, index):
        # 获取mri_nii中指定索引的路径
        mri_path = self.mri_nii[index]
        # 调用start_transformer函数，将mri_path作为参数传入，返回一个字典
        batch = self.start_transformer(dict(image=mri_path))
        # 对batch中的image进行自适应归一化
        # batch['image'] = adaptive_normal(batch['image'])
        # 调用transformer函数，将batch作为参数传入，返回一个字典
        batch = self.transformer(batch)
        # 将batch中的image的维度设置为1
        batch['image'] = batch["image"][:1,...]
        # 从mri_path中获取label
        batch['label'] = int(re.findall('-(\d).nii.gz', mri_path)[0])
        # 如果import_table为真，则从table_dict中获取cate_x和conti_x
        if self.import_table:
            date_index = self.find_index(mri_path)
            batch['cate_x'] = torch.tensor(self.table_dict['cate_x'].iloc[date_index].values, dtype=torch.float32).unsqueeze(-1)
            batch['conti_x'] = torch.tensor(self.table_dict['conti_x'].iloc[date_index].values, dtype=torch.float32).unsqueeze(-1)
        # 从mri_path中获取name
        batch['name'] = mri_path.split('/')[-1]
        # 返回batch
        return batch

    def find_index(self, mri_path):
        ID, date_time = re.search('.*/(.*?)-(.*?)_\d{2}_\d{2}_\d{2}.\d-\d.nii.gz', mri_path).groups()
        date_index = self.find_row(ID, date_time)
        if isinstance(date_index, tuple):
            # print(f'Cannot find id:{date_index[0]} with time:{date_index[1]} and label:{batch["label"]}')
            return None
        return date_index
    
    def __len__(self):
        return len(self.mri_nii)

class MRI_classify_2(data.Dataset):
    def __init__(self, data_path, table_path='', 
                 desired_shape=(160, 160, 96)):
        super(MRI_classify, self).__init__()
        # 获取data_path路径下的所有nii.gz文件
        self.mri_nii = glob(join(data_path, '*.nii.gz'))
        # 初始化加载器
        self.start_transformer = LoadImaged(keys=['image'])
        
        # 初始化转换器
        self.transformer = Compose(
            [
            # 将输入的nii.gz文件转换为3D张量
            EnsureChannelFirstd(keys=['image']),
            # 将输入的nii.gz文件转换为指定大小
            Resized(keys=['image'], spatial_size=desired_shape), 
            # 将输入的nii.gz文件 intensity range转换为指定范围
            ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1000, b_min=-1.0, b_max=1.0, clip=True),

            # 将转换后的nii.gz文件转换为tensor
            ToTensord(keys=['image'])
            ])
        # 判断table_path是否为空
        self.import_table = len(table_path)
        # 如果table_path不为空，则读取table_path中的csv文件
        if self.import_table:
            # self.table_dict = pd.read_csv(table_path)
            self.table_dict = prepare_table(table_path)
            # 遍历mri_nii中的每一个文件
            for i, path in enumerate(self.mri_nii):
                # 判断是否在table_dict中
                if self.find_index(mri_path=path) == None:
                    # 如果不在，则从mri_nii中删除该文件
                    self.mri_nii.pop(i)
            
        
    def find_row(self, ID, current_datetime):
        info = self.table_dict['info']
        min  = min_orig = 30
        min_index = -1
        difference = -1
        for index, row in info.iterrows():
            idInCsv, dateInCsv = row.iloc[0], row.iloc[1]
            if ID != idInCsv:
                continue
            difference = date_difference(dateInCsv, current_datetime)
            if min > difference:
                min = difference
                print(min)
                min_index = index

            if min <= 30 :
                break
        if min < min_orig :
            print("找到日期误差小于30天对应数据！")
            return min_index
        else:
            print("找不到日期误差小于30天"+" 对应数据(匹配数据信息：ID:"+str(ID)+"||date："+str(current_datetime)+")！")
            print(f"最小的日期差距是： {difference}")
            return (ID, current_datetime)

    def __getitem__(self, index):
        # 获取mri_nii中指定索引的路径
        mri_path = self.mri_nii[index]
        # 调用start_transformer函数，将mri_path作为参数传入，返回一个字典
        batch = self.start_transformer(dict(image=mri_path))
        # 对batch中的image进行自适应归一化
        # batch['image'] = adaptive_normal(batch['image'])
        # 调用transformer函数，将batch作为参数传入，返回一个字典
        batch = self.transformer(batch)
        # 将batch中的image的维度设置为1
        batch['image'] = batch["image"][:1,...]
        # 从mri_path中获取label
        batch['label'] = int(re.findall('-(\d).nii.gz', mri_path)[0])
        # 如果import_table为真，则从table_dict中获取cate_x和conti_x
        if self.import_table:
            date_index = self.find_index(mri_path)
            batch['cate_x'] = torch.tensor(self.table_dict['cate_x'].iloc[date_index].values, dtype=torch.float32).unsqueeze(-1)
            batch['conti_x'] = torch.tensor(self.table_dict['conti_x'].iloc[date_index].values, dtype=torch.float32).unsqueeze(-1)
        # 从mri_path中获取name
        batch['name'] = mri_path.split('/')[-1]
        # 返回batch
        return batch

    def find_index(self, mri_path):
        ID, date_time = re.search('.*/(.*?)-(.*?)_\d{2}_\d{2}_\d{2}.\d-\d.nii.gz', mri_path).groups()
        date_index = self.find_row(ID, date_time)
        if isinstance(date_index, tuple):
            # print(f'Cannot find id:{date_index[0]} with time:{date_index[1]} and label:{batch["label"]}')
            return None
        return date_index
    
    def __len__(self):
        return len(self.mri_nii)


def form_dataloader(updir, image_size, batch_size, shuffle=True):
    dataset = MRI2PET_dataset(updir, image_size)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True) 

def classi_dataloader(updir, image_size, batch_size, table_path, shuffle=True, **kwargs):
    dataset = MRI_classify(updir, table_path, image_size, **kwargs)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)

if __name__ == "__main__":
    import sys; sys.path.append('./')
    import time
    import torch
    from utils.common import see_mri_pet
    from torchvision.utils import save_image

    train_dataloader = form_dataloader('/data/publicData/CYFData/ZSH/Datasets/ADNI_MRI2PET/train', 
                                         (160, 160, 96), batch_size=1)
    start_time = time.time()
    batch = first(train_dataloader)
    end_time = time.time()
    print("Time: ", end_time - start_time)
    print("Shape: ", batch['image'].shape)
    image = batch['image']
    label = batch['label']
    image = torch.cat([image, label], dim=-2)
    # label = batch['label'][0, 0,...]
    save_image(see_mri_pet(image), 'combine.png')
    