import sys    #导入sys模块
import os
#from PIL import Image  #PIL是python的第三方图像处理库

#dcm2niix.exe -f "outputname" -i y -m y -p y -x y -z y -b n -o "E:\datasets\liver_ct_process_output" "E:\datasets\LiverCT\庄健忠\20180105000198"
# system command line
d2n = 'E:\PythonProject\pet2nii\MRIcroGL\Resources\dcm2niix.exe -f '
para = ' "%f_%p_%t_%s" -p y -z y -o '
output_path = "/liver_ct_process_output"

###产生随机命名
import random

def ranstr(num):
    #dictionary
    H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

    salt = ''
    for i in range(num):
        salt += random.choice(H)

    return salt

# read txt
filepaths = []  #Input dicom file directory
for line in open("process_output/donelist.txt", "r"):  # Setting up the file object and reading each line of the file
    filepaths.append(line)
# deal with format transform

out_dir = os.path.join('/', 'ADNI')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

output_path = 'E:\PythonProject\MRIGet\Dataset\ADNI_3year\ADNI_1'
for filepath in filepaths:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filepath = filepath[:-1]
    outputname = os.path.basename(filepath)
    cmd = d2n + outputname + para +'\"'+ output_path +'\" ' + '\"'+ filepath +'\"'
    os.system(cmd)

print("Congratulation, Done!")


