# **Dataset Acquisition**

## **1. Saving Data with `get_all_sample`**

Save all data from the table into the ADNI folder for subsequent operations using 
```
get_all_sample\get_all.py
```

## **2. Filtering and Saving Paired Pretrained MRI-PET Data**

Filter paired pretrained MRI-PET data using 
```
pretrain_MRI-PET\find_mri-pet.py
```
then save the data into the ADNI web folder using 
```
pretrain_MRI-PET\get_mri-pet.py
```
 
## **3. Filtering and Saving Training Data**

Filter the required training data tables using 
```
train_data\filt&copy_MCI2AD.py
```
```
train_data\filt&copy_neg.py
```
Read the data tables and save the 1-year dataset and 3-year dataset into the ADNI folder using 
```
train_data\dir_find.py
```

## **4. Processing and Converting Dataset Format**

Download the ADNI website folder locally, process the original folder format using 
```
dcm2nii\processing.py
```
Then convert it to the final `nii.gz` dataset format using 
```
dcm2nii\2txt.py
```
```
dcm2nii\2nii.py
```

## **Notes**

1. You need to fill in your ADNI account and password when using the script.
2. If the program cannot run normally due to network issues, you need to manually select the ADNI dataset web page and folder.
3. You can get 'TADPOLE_D1_D2.csv' from 'https://drive.google.com/file/d/1K6d0bv-A6Fzoqz9LfECM81MJTYIpKbGV/view?usp=sharing'
