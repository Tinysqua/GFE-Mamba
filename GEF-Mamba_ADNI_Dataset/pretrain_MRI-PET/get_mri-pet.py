from datetime import datetime
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from time import sleep
from datetime import datetime

def logging(driver):
    driver.get('https://ida.loni.usc.edu/pages/access/search.jsp')
    sleep(2)
    try:
        accept = driver.find_element(By.CSS_SELECTOR, '.ida-cookie-policy-right > div:nth-child(1)')
        accept.click()
    except NoSuchElementException:
        print("Element not found！")

    sleep(3)
    log = driver.find_element(By.CSS_SELECTOR, 'div.ida-menu-option:nth-child(4) > div:nth-child(1)')
    log.click()

    email_input = driver.find_element(By.CSS_SELECTOR, '.ida-menu-email-container > input:nth-child(2)')
    email_input.send_keys('https://ida.loni.usc.edu/pages/access/search.jsp')

    password_input = driver.find_element(By.CSS_SELECTOR, '.ida-menu-password-container > input:nth-child(3)')
    password_input.send_keys('')

    logging = driver.find_element(By.CSS_SELECTOR, '.login-btn > span:nth-child(2)')
    logging.click()

def calculate_date_difference(date1, date2):
    date_format = "%m/%d/%Y"
    datetime_object1 = datetime.strptime(date1, date_format)
    datetime_object2 = datetime.strptime(date2, date_format)

    difference = abs(datetime_object2 - datetime_object1)

    return difference.days

import pandas as pd

df = pd.read_csv('demo_1_31_2024.csv')

mri_dict = {}
pet_dict = {}

from collections import defaultdict

# 初始化字典
mri_dict = defaultdict(list)
pet_dict = defaultdict(list)
result_dict = defaultdict(list)

for index, row in df.iterrows():
    id = row['Subject']
    modality = row['Modality']
    date = row['Acq Date']
    if modality == 'MRI':
        mri_dict[id].append(date)
    elif modality == 'PET':
        pet_dict[id].append(date)

for id in mri_dict.keys():
    if id in pet_dict:
        for mri_date in mri_dict[id]:
            for pet_date in pet_dict[id]:
                diff = calculate_date_difference(mri_date, pet_date) / 30
                if diff < 5:
                    result_dict[id].append((mri_date, pet_date))

# 初始化字典
pet_dict = defaultdict(list)
mri_dict = defaultdict(list)

# 遍历结果字典
for id, date_pairs in result_dict.items():
    for mri_date, pet_date in date_pairs:
        pet_dict[id].append(pet_date)
        mri_dict[id].append(mri_date)

total_pairs = sum(len(v) for v in result_dict.values())
print(str(result_dict))
print(str(total_pairs))

# 遍历数据
for index, row in df.iterrows():
    id = row['Subject']
    modality = row['Modality']
    date = row['Acq Date']
    if modality == 'MRI':
        if id not in mri_dict:
            mri_dict[id] = []
        mri_dict[id].append(date)
    elif modality == 'PET':
        if id not in pet_dict:
            pet_dict[id] = []
        pet_dict[id].append(date)


for id in list(mri_dict.keys()):
    if id in pet_dict:
        min_diff = float('inf')
        best_dates = None
        for mri_date in mri_dict[id]:
            for pet_date in pet_dict[id]:
                diff = calculate_date_difference(mri_date, pet_date) / 30
                if diff < 5 and diff < min_diff:
                    min_diff = diff
                    best_dates = (mri_date, pet_date)
        if best_dates is not None:
            mri_dict[id], pet_dict[id] = best_dates
        else:
            mri_dict[id] = pet_dict[id] = None
    else:
        del mri_dict[id]

# Now, the mri_dict may also contain some keys that don't exist in the pet_dict, so let's iterate through it again and remove those keys
for id in list(pet_dict.keys()):
    if id not in mri_dict:
        del pet_dict[id]

print('--------------------------------------------------------------------------\n\n')
print('MRI:', mri_dict)
print('--------------------------------------------------------------------------\n\n')
print('PET:', pet_dict)
print(len(mri_dict))
print(len(pet_dict))


while True:
        driver = webdriver.Firefox()
        logging(driver)

        # Manual selection to the collection page

        table = driver.find_element(By.CSS_SELECTOR,'#tableData > table:nth-child(2)')
        rows = table.find_elements(By.TAG_NAME, 'tr')

        for row in rows:
            while True:
                try:
                    cols = row.find_elements(By.TAG_NAME, 'td')
                    break
                except Exception as e:
                    sleep(2)
            if cols[5].text == 'PET':
                if cols[0].text in pet_dict:
                    if cols[8].text == pet_dict[cols[0].text]:
                        cols[11].find_elements(By.TAG_NAME, 'input')[1].click()
            elif cols[0].text in mri_dict and cols[8].text == mri_dict[cols[0].text]:
                cols[11].find_elements(By.TAG_NAME, 'input')[1].click()

        slider = driver.find_element(By.CSS_SELECTOR,'#collection_table > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > table:nth-child(2) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(3) > td:nth-child(1) > input:nth-child(1)')
        slider.click()
        sleep(1)

import csv

with open('demo_1_31_2024.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    totol = 0
    for row in reader:
        if row[6] == 'PET':
            if row[1] in pet_dict:
                if row[9] in pet_dict[row[1]]:
                    with open('final_1_31_2024.csv', 'r') as f_final:
                        reader_final = csv.reader(f_final)
                        next(reader_final)
                        bol = False
                        for row_final in reader_final:
                            if row[1] == row_final[1] and row[9] == row_final[9]:
                                bol = True
                        if bol == False:
                            print('lost pet'+str(row[1])+'||'+str(row[9]))

                    totol += 1
        elif row[1] in mri_dict and row[9] in mri_dict[row[1]]:
            with open('final_1_31_2024.csv', 'r') as f_final:
                reader_final = csv.reader(f_final)
                next(reader_final)
                bol = False
                for row_final in reader_final:
                    if row[1] == row_final[1] and row[9] == row_final[9]:
                        bol = True
                if bol == False:
                    print('lost mri' + str(row[1]) + '||' + str(row[9]))
            totol += 1
    print(totol)


