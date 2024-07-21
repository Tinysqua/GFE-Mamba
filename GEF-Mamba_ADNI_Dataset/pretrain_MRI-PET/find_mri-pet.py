import os
import shutil
import pandas as pd
from datetime import datetime

def find_folder(path, first_layer_folder, target_folder):
    for root, dirs, files in os.walk(path):
        if first_layer_folder in root:
            for name in dirs:
                if name == target_folder:
                    return os.path.join(root, name)
    return None

def calculate_date_difference(date1, date2):

    date_format = "%m/%d/%Y"
    datetime_object1 = datetime.strptime(date1, date_format)
    datetime_object2 = datetime.strptime(date2, date_format)

    difference = abs(datetime_object2 - datetime_object1)

    return difference.days


df = pd.read_csv('final_1_31_2024.csv')

mri_dict = {}
pet_dict = {}

from collections import defaultdict

mri_dict = defaultdict(list)
pet_dict = defaultdict(list)
result_dict = defaultdict(list)
ID_dic = defaultdict(list)
ID_pairs_dic = defaultdict(list)

for index, row in df.iterrows():
    ID = row['Image Data ID']
    subject = row['Subject']
    modality = row['Modality']
    date = row['Acq Date']
    if modality == 'MRI':
        mri_dict[subject].append(date)
    elif modality == 'PET':
        pet_dict[subject].append(date)
    ID_dic[(subject,modality,date)] = ID

# Iterate over each key in the dictionary
for subject in mri_dict.keys():
    # If this key exists in both dictionaries
    if subject in pet_dict:
        for mri_date in mri_dict[subject]:
            for pet_date in pet_dict[subject]:
                # Calculate the difference in days between two dates using the function provided, then convert to a difference in months
                diff = calculate_date_difference(mri_date, pet_date) / 30
                # If the difference in months is less than 5
                if diff < 5:
                    # Add date pairs to the result dictionary
                    result_dict[subject].append((mri_date, pet_date))
                    ID_pairs_dic[subject].append((str(ID_dic[(subject,'MRI',mri_date)]),str(ID_dic[(subject,'PET',pet_date)])))
print(result_dict['941_S_5193'])
print(ID_dic['941_S_5193','MRI','5/29/2013'])
print(ID_pairs_dic)
# - Suppose your folder structure is as follows:
# - Under the root directory (root_dir) there are a number of numbered folders.
# - under each numbered folder there are two subfolders: 'Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution' and 'MP-RAGE'
# - under these two subfolders there are further folders named after dates
root_dir = 'E:\Downloads\ADNI'
target_dir = 'E:\PythonProject\MedDataGet\ADNI__'

for subject, ID_pairs in ID_pairs_dic.items():
    total = 0
    for ID_pair in ID_pairs:
        print(ID_pair)
        target = os.path.join(target_dir, subject + '-' + str(total))
        total += 1
        for i, ID in enumerate(ID_pair):
            folder_path = find_folder(root_dir, subject, ID)
            print(folder_path)
            if folder_path:
                print("Found the folder.：", folder_path)
                ID_path = os.path.join(target,'MRI') if i == 0 else os.path.join(target,'PET')
                if not os.path.exists(ID_path):
                    shutil.copytree(folder_path, ID_path)



