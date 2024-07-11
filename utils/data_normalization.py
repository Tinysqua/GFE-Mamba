import os
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
import nibabel as nib
import numpy as np
import torch
from os.path import join


 
import os
def adaptive_normal(img):
 
    min_p = 0.001 
    max_p = 0.999 # quantile prefer 98~99
    
    imgArray = img
    imgPixel = imgArray[imgArray >= 0]
    imgPixel, _ = torch.sort(imgPixel)
    index = int(round(len(imgPixel) - 1) * min_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_min = imgPixel[index]
 
    index = int(round(len(imgPixel) - 1) * max_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_max = imgPixel[index]
 
    mean = (value_max + value_min) / 2.0
    stddev = (value_max - value_min) / 2.0
    imgArray = (imgArray - mean) / stddev
    imgArray[imgArray < -1] = -1.0
    imgArray[imgArray > 1] = 1.0
 
    return imgArray

if __name__ == "__main__":
    data_dir = 'datasets/ADNI_dataset'
    train_sub_dir = sorted(os.listdir(data_dir))
    print("train_sub_dir", train_sub_dir)
    train_files = [{"image": join(data_dir, i, 'mri.nii.gz'), 'label': join(data_dir, i, 'pet.nii.gz')} for i in train_sub_dir]

    start_transformer = Compose([LoadImaged(keys=['image', 'label']), 
                                EnsureChannelFirstd(keys=['image', 'label']),
                                CropForegroundd(keys=['label'], source_key='label'),
                                ])
    train_file = train_files[0]
    # result = {}
    # result['image'] = nib.load(train_file['image']).get_fdata()
    # result['label'] = nib.load(train_file['label']).get_fdata()
    result = start_transformer(train_file)

    print("Min image: ", result['image'].min())
    print("Min label: ", result['label'].min())

    result['image'] = adaptive_normal(result['image'])
    result['label'] = adaptive_normal(result['label'])

    print("Max image: ", torch.min(result['image']))
    print("Max label: ", torch.min(result['label']))

    print(result['image'].shape)