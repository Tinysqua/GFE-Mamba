# **DATASET ACQUISITION**

## **1. SAVING DATA WITH `get_all_sample`**

Save all data from the table into the ADNI folder for subsequent operations using `get_all_sample\get_all.py`.

## **2. FILTERING AND SAVING PAIRED PRETRAINED MRI-PET DATA**

Filter paired pretrained MRI-PET data using `pretrain_MRI-PET\find_mri-pet.py` in `pretrain_MRI-PET\pretrain_MRI-PET`, then save the data into the ADNI web folder using `pretrain_MRI-PET\get_mri-pet.py`.

## **3. FILTERING AND SAVING TRAINING DATA**

Filter the required training data tables using `train_data\filt&copy_MCI2AD.py` and `train_data\filt&copy_neg.py`. Read the data tables using `train_data\dir_find.py` and save the 1-year dataset and 3-year dataset into the ADNI folder.

## **4. PROCESSING AND CONVERTING DATASET FORMAT**

Download the ADNI website folder locally, process the original folder format using `dcm2nii\processing.py`, then convert it to the final `nii.gz` dataset format using `dcm2nii\2txt.py` and `dcm2nii\2nii.py`.

## **NOTES**

1. You need to fill in your ADNI account and password when using the script.
2. If the program cannot run normally due to network issues, you need to manually select the ADNI dataset web page and folder.
