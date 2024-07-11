import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
sns.set_style('darkgrid')
from scipy.ndimage import zoom



def read_nii(img_path, dir_path, index):
    img = nib.load(img_path)

    # 获取图像数据
    data = img.get_fdata()
    # data =  np.rot90(data)
    data = data[...,1]
    data = zoom(data, (1, 1, 1))
    # data = np.swapaxes(data, 0, 2)
    # data = np.swapaxes(data, 0, 1)
    print("The shape of data: ", data.shape)
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

    # plt.show()
    plt.savefig(os.path.join(dir_path, f'slicess_{index}.png'))    
    
if __name__ == '__main__':
    current_dir = r"datasets/ADNI/094_S_1164-1"
    files = os.listdir(current_dir)
    for i, file in enumerate(files):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            read_nii(os.path.join(current_dir, file), current_dir, index=i)

