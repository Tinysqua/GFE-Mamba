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

mri = r'datasets/ADNI/094_S_1164-1/mri.nii.gz'
trans = LoadImaged(keys=['image'])
a = trans(dict(image=mri))
print(a['image'].shape)