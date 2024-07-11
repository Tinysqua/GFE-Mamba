import pandas as pd
from sklearn.calibration import LabelEncoder
import re

from sklearn.discriminant_analysis import StandardScaler
def has_letters(string):
    if not isinstance(string, str):
        return False
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    if match:
        return True
    else:
        return False
    
def discovery_mix(df):
    str_columns = df.select_dtypes(include='object').columns
    # 检查包含字符串类型的列是否具有混合类型
    mixed_columns = []
    for column in str_columns:
        if df[column].apply(has_letters).sum() > 0:
            mixed_columns.append(column)
    # 打印含有混合类型的列
    print("混合类型的列：", mixed_columns)
    return mixed_columns


def prepare_table(mri_df):
    # mri_df = pd.read_csv(table_path)
    drop_list = ['RID', 'D2', 'SITE', 'DX', 'COLPROT', 'ORIGPROT', 'Month', 
                 'M', 'FDG', 'PIB', 'AV45']
    info_list = ['PTID', 'EXAMDATE', 'LABEL']
    for i, col in enumerate(mri_df.columns):
        if 'bl' in col:
            drop_list.append(col)
    table_info = mri_df[info_list]
    mri_df = mri_df.drop(drop_list + info_list, axis=1)
    mixed_columns = discovery_mix(mri_df)
    num_columns = [x for x in mri_df.columns if x not in mixed_columns]
    mri_df[mixed_columns] = mri_df[mixed_columns].fillna('NA').astype('category')
    num_cat = []
    for col in mri_df.columns:
        if col in mixed_columns:
            mri_df[col] = LabelEncoder().fit_transform(mri_df[col])
            num_cat.append(len(mri_df[col].unique()))
            # mri_df[col] = LabelEncoder().fit_transform(mri_df[col])
        else:
            mri_df[col] = pd.to_numeric(mri_df[col], errors='coerce')
            mri_df[col] = mri_df[col].fillna(0)
    sc = StandardScaler()
    sc.fit(mri_df[num_columns])
    mri_df[num_columns] = sc.transform(mri_df[num_columns])

    dfcats = mri_df[mixed_columns]
    df_categorical_encoded = dfcats
    # df_categorical_encoded = pd.get_dummies(dfcats)

    dfconts = mri_df.drop(mixed_columns, axis=1)
    return_dict = {"info": table_info, "cate_x": df_categorical_encoded,"conti_x": dfconts, 
                   "num_cat": num_cat, "num_cont": len(num_columns)}
    return return_dict

