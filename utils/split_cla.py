import os
import random
import shutil

from tqdm import tqdm

folder_path = "/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_3year"
nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".nii.gz")]

test_size = int(0.2 * len(nii_files))
test_files = random.sample(nii_files, test_size)
train_files = [f for f in nii_files if f not in test_files]

train_folder = "/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_3year_train"
test_folder = "/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_3year_test"

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)



# 复制训练集文件
for file in tqdm(train_files, desc="Copying training files"):
    shutil.copy(file, train_folder)

# 复制测试集文件
for file in tqdm(test_files, desc="Copying testing files"):
    shutil.copy(file, test_folder)