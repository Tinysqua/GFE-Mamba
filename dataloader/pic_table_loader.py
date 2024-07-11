import sys; sys.path.append('./')
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils import data
import torch
from os.path import join
from glob import glob
import re
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
from utils.data_normalization import adaptive_normal
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


class MRI_classify(data.Dataset):
    def __init__(self, data_path, table_path='', 
                 desired_shape=(160, 160, 96), days_threshold=-1):
        super(MRI_classify, self).__init__()
        self.mri_nii = glob(join(data_path, '*.nii.gz'))
        self.start_transformer = LoadImaged(keys=['image'])
        
        self.transformer = Compose(
            [
            EnsureChannelFirstd(keys=['image']),
            Resized(keys=['image'], spatial_size=desired_shape), 
            # ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1000, b_min=-1.0, b_max=1.0, clip=True),

            ToTensord(keys=['image'])
            ])
        self.import_table = len(table_path)
        if self.import_table:
            self.table_df = pd.read_csv(table_path)
            # self.table_dict = prepare_table(table_path)
            print(f"Num before filter: {len(self.mri_nii)}")
            for i, path in enumerate(self.mri_nii):
                search_result = self.find_index(mri_path=path.split('/')[-1], to_find_table=self.table_df)
                min_index = search_result[1]
                if search_result[0] == False:
                    self.mri_nii.pop(i)
                if self.table_df.iloc[min_index]['date_diff'] <= days_threshold:
                    # print(f'{ID} with {to_find_table.iloc[min_index]["LABEL"]} date_diff too small: {to_find_table.iloc[min_index]["date_diff"]}')
                    self.mri_nii.pop(i)
            self.table_df = prepare_table(self.table_df)
            print(f"Num after filter: {len(self.mri_nii)}")
            
        
    def find_row(self, ID, current_datetime, ischanged, to_find_table):
        subset = to_find_table[(to_find_table['PTID'] == ID)]
        min = 31
        min_index = -1
        for index, data in subset.iterrows():
            dateInCsv = data['EXAMDATE']
            # print(dateInCsv + ' ' + str(data[10]) + ' ' +  str(ischanged))
            if (pd.isna(data['LABEL']) == False and (
                    (ischanged == '1' and  int(data['LABEL']) == 1 ) or (ischanged == '0' and int(data['LABEL']) == 0))):
                if min > date_difference(dateInCsv, current_datetime):
                    min = date_difference(dateInCsv, current_datetime)
                    min_index = index

            if min == 0:
                break
        # if to_find_table.iloc[min_index]['date_diff'] <= 30:
        #     print(f'{ID} with {to_find_table.iloc[min_index]["LABEL"]} date_diff too small: {to_find_table.iloc[min_index]["date_diff"]}')
        #     return (False, min_index)
        
        if min != 31:
            return (True, min_index)
        else:
            print("找不到日期误差小于30天ischanged= " + str(ischanged) + " 对应数据(匹配数据信息：ID:" + str(
                ID) + "||date：" + str(current_datetime) + ")！")
            return (False, min_index) # index should be -1

    def __getitem__(self, index):
        mri_path = self.mri_nii[index]
        # print("Name: ", mri_path.split('/')[-1])
        batch = self.start_transformer(dict(image=mri_path))
        batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1,...]
        batch['label'] = int(re.findall('-(\d).nii.gz', mri_path)[0])
        if self.import_table:
            _, date_index = self.find_index(mri_path.split('/')[-1], self.table_df['info'])
            batch['cate_x'] = torch.tensor(self.table_df['cate_x'].iloc[date_index].values, dtype=torch.int64)
            batch['conti_x'] = torch.tensor(self.table_df['conti_x'].iloc[date_index].values, dtype=torch.float32)
        batch['name'] = mri_path.split('/')[-1]
        return batch

    def find_index(self, mri_path, to_find_table=None):
        ID, date , ischanged = mri_path.split('-')
        ischanged = str(ischanged.split('.')[0])
        date = date.split('_')[0] + '-' + date.split('_')[1] + '-' + date.split('_')[2]
        status, min_index = self.find_row(ID, date, ischanged, to_find_table)
        return (status, min_index)
    
    def __len__(self):
        return len(self.mri_nii)



def classi_dataloader(updir, image_size, batch_size, table_path, shuffle=True, **kwargs):
    dataset = MRI_classify(updir, table_path, image_size, **kwargs)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)

if __name__ == "__main__":
    import sys; sys.path.append('./')
    import time
    import torch
    from utils.common import see_mri_pet
    from torchvision.utils import save_image

    train_dataloader = classi_dataloader('/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_3year_train', 
                                         (160, 160, 96), batch_size=16, 
                                         table_path='/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ct_2&5_3year.csv')
    start_time = time.time()
    batch = first(train_dataloader)
    end_time = time.time()
    print("Time: ", end_time - start_time)
    print("Shape: ", batch['image'].shape)
    image = batch['image']
    # label = batch['label'][0, 0,...]
    save_image(see_mri_pet(image), 'combine.png')
    # plt_mri_pet(torch.cat((image, label), dim=-2), 'combine.png')
    