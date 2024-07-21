import os
import shutil

# 源文件夹路径
src_dir = 'E:\PythonProject\pet2nii\positive_samples_process\ADNI'
# 目标文件夹路径
dst_dir = 'E:\PythonProject\pet2nii\dcm2nii\process02\ADNI_pos'  # 请替换为你的目标文件夹路径

# 遍历源文件夹下的所有子文件夹
for dirpath, dirnames, filenames in os.walk(src_dir):
    for dirname in dirnames:
        if dirname.startswith('I'):
            subfolder_path = os.path.join(dirpath, dirname)
            # 使用os库的split函数分割文件路径
            path_parts = os.path.normpath(subfolder_path).split(os.sep)

            code = path_parts[-4]
            date = path_parts[-2].replace('-','_')
            name = code + '-' + date + '-' + '1'

            dst_folder_path = os.path.join(dst_dir, name)
            if not os.path.exists(dst_folder_path):
                shutil.copytree(subfolder_path, dst_folder_path)

                print(f'已将 {subfolder_path} 复制到 {dst_folder_path} 并更改文件名为 {name}')
            else:
                print(dst_folder_path + ' 已存在')


        # 使用os库的join函数连接目录路径和子目录名，得到子目录的完整路径
        # # 从文件夹名中提取编号和日期
        # code, date = folder_name.split('\\')[1], folder_name.split('\\')[3]
        # # 创建新的文件名
        # new_folder_name = f"{code}-{date.replace('-', '_')}-1"
        # # 创建目标文件夹路径
        # dst_folder_path = os.path.join(dst_dir, new_folder_name)
        # # 复制文件夹
        # shutil.copytree(src_folder_path, dst_folder_path)
        # print(f'已将 {src_folder_path} 复制到 {dst_folder_path} 并更改文件名为 {new_folder_name}')
