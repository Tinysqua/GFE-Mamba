from datetime import datetime
import yaml
import inspect
from shutil import copyfile, copy
import os
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_plot_data(epoch:int, predictions:torch.Tensor, targets:torch.Tensor, parrent_dir:str):
    '''
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    '''
    save_data = {
        'epoch': epoch,
        'predictions': predictions,  # 堆叠为一个大数组
        'targets': targets,
    }
    torch.save(save_data, f'{parrent_dir}/epoch_{epoch}_data.pth')

def date_difference(date1, date2):
    # 将日期字符串转换为datetime对象
    date_format1 = '%Y-%m-%d'
    date_format2 = '%Y-%m-%d'

    datetime_object1 = datetime.strptime(date1, date_format1)
    datetime_object2 = datetime.strptime(date2, date_format2)

    # 计算两个日期之间的差异
    difference = abs(datetime_object2 - datetime_object1)

    # 返回差异的天数
    return difference.days

def see_mri_pet(tensor_3d, normalize=True):
    # new_size = (tensor_3d.shape[2]//2, tensor_3d.shape[3], tensor_3d.shape[4])
    # resizer = transforms.Resize(new_size)
    # tensor_3d = resizer(tensor_3d)
    tensor_3d = tensor_3d[0,0,...] # take away the channel and batch dim
    tensor_3d = tensor_3d.permute(2, 0, 1)
    pic = make_grid(tensor_3d.unsqueeze(1))
    if normalize:
        pic = (pic+1)/2
    else:
        pic = pic
    return pic

def plt_mri_pet(data, save_path):
    # print("The shape of data: ", data.shape)
    # 可视化每一层切片
    num_slices = data.shape[-1]

    # 设置子图的行数和列数
    num_rows = num_slices // 10 + 1  # 每行显示10个切片
    num_cols = min(num_slices, 10)

    # 设置子图的大小
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # 遍历每一层切片并可视化
    for i in range(num_slices):
        row_idx = i // 10
        col_idx = i % 10

        # 在子图中显示每一层切片
        axes[row_idx, col_idx].imshow(data[:, :, i], cmap='gray')
        axes[row_idx, col_idx].axis('off')  # 关闭坐标轴

    # 如果切片数量不是10的倍数，隐藏多余的子图
    for i in range(num_slices, num_rows * num_cols):
        row_idx = i // 10
        col_idx = i % 10
        fig.delaxes(axes[row_idx, col_idx])

    plt.savefig(save_path)  

def copy_yaml_to_folder(yaml_file, folder):
    """
    将一个 YAML 文件复制到一个文件夹中
    :param yaml_file: YAML 文件的路径
    :param folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(folder, exist_ok=True)

    # 获取 YAML 文件的文件名
    file_name = os.path.basename(yaml_file)

    # 将 YAML 文件复制到目标文件夹中
    copy(yaml_file, os.path.join(folder, file_name))

def copy_yaml_to_folder_auto(yaml_file, folder):
    """
    将一个 YAML 文件复制到一个文件夹中
    :param yaml_file: YAML 文件的路径
    :param folder: 目标文件夹路径
    """

    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%m%d%H%M%S")

    #获取运行程序名
    program_name_with_ext = os.path.basename(sys.argv[0])
    program_name, ext = os.path.splitext(program_name_with_ext)

    # 确保目标文件夹存在
    dir = os.path.join(folder, os.path.basename('exp_' + str(formatted_time) + '_' + program_name))
    os.makedirs(dir, exist_ok=True)

    # 获取 YAML 文件的文件名
    file_name = os.path.basename(yaml_file)

    # 将 YAML 文件复制到目标文件夹中
    copy(yaml_file, os.path.join(dir, file_name))

    return dir

# 加载配置文件
def load_config(file_path):
    # 打开文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 加载文件
        config = yaml.safe_load(f)
        # 遍历配置文件中的每一项
        for key in config.keys():
            # 如果该项的值是列表，则将其转换为元组
            if type(config[key]) == list:
                config[key] = tuple(config[key])
        # 返回配置文件
        return config
    
def get_parameters(fn, original_dict):
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict

def write_config(config_path, save_path):
    copyfile(config_path, save_path)
    
