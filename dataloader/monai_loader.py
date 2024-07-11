import os
from glob import glob
import sys; sys.path.append('./')
from utils.common import see_mri_pet, plt_mri_pet
from torchvision.utils import save_image
import torch
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

from monai.data import Dataset, DataLoader
from monai.utils import first
import matplotlib.pyplot as plt
import os
from os.path import join

data_dir = 'datasets/ADNI_dataset'
train_sub_dir = sorted(os.listdir(data_dir))
print("train_sub_dir", train_sub_dir)
# val_images = sorted(glob(os.path.join(data_dir, 'ValData', '*.nii.gz')))
# val_labels = sorted(glob(os.path.join(data_dir, 'ValLabels', '*.nii.gz')))

train_files = [{"image": join(data_dir, i, 'mri.nii.gz'), 'label': join(data_dir, i, 'pet.nii.gz')} for i in train_sub_dir]
# val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]


# load the images
# do any transforms
# need to convert them into torch tensors

orig_transforms = Compose(

    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Resized(keys=['image', 'label'], spatial_size=[512,512,96]), 
        ToTensord(keys=['image', 'label'])
    ]
)

train_transforms = Compose(

    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
        ScaleIntensityRanged(keys=['image', 'label'], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(keys=['image', 'label'], spatial_size=[128,128,96]),
        ToTensord(keys=['image', 'label'])
    ]
)

demo_transforms = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        CropForegroundd(keys=['label'], source_key='label'),
        Resized(keys=['image', 'label'], spatial_size=[160,160,96]), 
        ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1203.0, b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=['label'], a_min=0.0, a_max=2.02, b_min=-1.0, b_max=1.0, clip=True),

        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2), mode=("bilinear")),
        ToTensord(keys=['image', 'label'])
    ]
)



orig_ds = Dataset(data=train_files, transform=orig_transforms)
orig_loader = DataLoader(orig_ds, batch_size=1)

train_ds = Dataset(data=train_files, transform=demo_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

# val_ds = Dataset(data=val_files, transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=1)
mri_max_value = 0
mri_min_value = 0
pet_max_value = 0
pet_min_value = 0
for i, batch in enumerate(orig_loader):
    mri_max_value += torch.max(batch["image"])
    print(f'The {i} with {torch.max(batch["image"])}')
    mri_min_value += torch.min(batch["image"])
    pet_max_value += torch.max(batch["label"])
    pet_min_value += torch.min(batch["label"])
    
    plt.figure('test', (12, 6))
    channel = 70
    plt.subplot(1, 3, 1)
    plt.title('Orig patient')
    plt.imshow(batch['image'][0, 0, : ,: ,channel], cmap= "gray")

    plt.subplot(1, 3, 2)
    plt.title('Slice of a patient')
    plt.imshow(batch['image'][0, 0, : ,: ,channel], cmap= "gray")

    plt.subplot(1,3,3)
    plt.title('Label of a patient')
    plt.imshow(batch['label'][0, 0, : ,: ,channel], cmap='gray')
    plt.savefig(f'test{i}.png')
    
avg_mri_max = mri_max_value / len(orig_loader)
avg_mri_min = mri_min_value / len(orig_loader)
avg_pet_max = pet_max_value / len(orig_loader)
avg_pet_min = pet_min_value / len(orig_loader)
print(f"MRI max: {avg_mri_max}, MRI min: {avg_mri_min}")
print(f"PET max: {avg_pet_max}, PET min: {avg_pet_min}")

test_patient = first(train_loader)
orig_patient = first(orig_loader)

print(torch.min(test_patient['image']))
print(torch.max(test_patient['image']))




save_image(see_mri_pet(test_patient['image'], False), 'test_mri.png')
save_image(see_mri_pet(test_patient['label'], False), 'test_pet.png')
# plt_mri_pet(test_patient['image'][0, 0,...], 'test_mri.png')
# plt_mri_pet(test_patient['label'][0, 0,...], 'test_pet.png')