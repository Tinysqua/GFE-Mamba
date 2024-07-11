import os
import shutil

import pandas as pd
from datetime import datetime
import csv

def date_difference(date1, date2):
    # 将日期字符串转换为datetime对象
    date_format1 = '%Y-%m-%d'
    date_format2 = '%Y_%m_%d'

    datetime_object1 = datetime.strptime(date1, date_format1)
    datetime_object2 = datetime.strptime(date2, date_format2)

    # 计算两个日期之间的差异
    difference = abs(datetime_object2 - datetime_object1)

    # 返回差异的天数
    return difference.days

import os

import pandas as pd
from datetime import datetime

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

def filt_csv(directory, csv_file, output_csv):

    totoal = 0
    # 读取csv文件
    df = pd.read_csv(csv_file)
    # 读取csv文件的头部
    output_df = pd.read_csv(csv_file, nrows=0)

    for filename in os.listdir(directory):
        totoal += 1

        ID, date , ischanged = filename.split('-')
        ischanged = str(ischanged.split('.')[0])
        date = date.split('_')[0] + '-' + date.split('_')[1] + '-' + date.split('_')[2]

        subset = df[(df['PTID'] == ID)]
        # 判断日期误差是否在正负30天内
        min = 31
        # 遍历每一行数据
        for index, data in subset.iterrows():
            dateInCsv = data['EXAMDATE']
            print(dateInCsv + ' ' + str(data[10]) + ' ' +  str(ischanged))
            if (pd.isna(data['DXCHANGE']) == False and (
                    (ischanged == '1' and  int(data['DXCHANGE']) == 5 ) or (ischanged == '0' and 1 <= int(data['DXCHANGE']) <= 3))):
                if min > date_difference(dateInCsv, date):
                    min = date_difference(dateInCsv, date)
                    minData = data

            if min == 0:
                break

        if min != 31:
            # print("找到日期误差小于30天对应数据！")
            pass
            # output_df = output_df._append(minData)

            # Dataset = 'E:\PythonProject\MRIGet\Dataset'
            # if not os.path.exists(os.path.join(Dataset,'ADNI_MRI2AD')):
            #     shutil.copy(os.path.join(directory,filename), os.path.join(Dataset,'ADNI_MRI2AD'))

        else:
            print("找不到日期误差小于30天ischanged= " + str(ischanged) + " 对应数据(匹配数据信息：ID:" + str(
                ID) + "||date：" + str(date) + ")！")

    # output_df.to_csv(output_csv, index=False)
    print(totoal)


directory = '/home/chenyifei/data/CYFData/FZJ/ADNI_MRI2AD&neg/ADNI_classifier_train_eval/train'

filt_csv(directory,'/home/chenyifei/data/CYFData/FZJ/ADNI_MRI2AD&neg/filt.csv', 'filt_MCI2AD.csv')
