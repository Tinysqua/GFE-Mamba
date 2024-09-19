# **Dataset Acquisition**

## **1. Saving Data in 'TADPOLE_D1_D2.csv'`**

Save all data from the table into the ADNI folder for subsequent operations using 
```
get_all_sample\get_all.py
```

## **2. Filtering and Saving Paired Pretrained MRI-PET Data**

Filter paired pretrained MRI-PET data using 
```
pretrain_MRI-PET\find_mri-pet.py
```
Then save the data into the ADNI web folder using 
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
3. You can get 'TADPOLE_D1_D2.csv' from [baidu disk](https://pan.baidu.com/s/1Sb-WBllct5RkEOnfAswANg?pwd=32sd) then put the csv file into 'get_all_sample' and 'train_data'.
