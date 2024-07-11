import shutil
import os
import random

def split_dataset(dataset_path, to_folder, train_ratio):
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    random.shuffle(subfolders)
    num_train = int(len(subfolders) * train_ratio)
    train_set = subfolders[:num_train]
    test_set = subfolders[num_train:]
    train_folder = os.path.join(to_folder, "train")
    test_folder = os.path.join(to_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    # 复制训练集文件夹
    for folder in train_set:
        folder_name = os.path.basename(folder)
        destination = os.path.join(train_folder, folder_name)
        shutil.copytree(folder, destination)
    # 复制测试集文件夹
    for folder in test_set:
        folder_name = os.path.basename(folder)
        destination = os.path.join(test_folder, folder_name)
        shutil.copytree(folder, destination)
    print("数据集已成功分割为训练集和测试集。")
# 指定数据集文件夹路径和训练集比例
dataset_folder = "datasets/ADNI"
to_folder = 'datasets/ADNI_train_eval'
train_ratio = 0.8
split_dataset(dataset_folder, to_folder, train_ratio)