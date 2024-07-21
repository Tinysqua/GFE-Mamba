from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from time import sleep
import pandas as pd
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def logging(driver):
    driver.get('https://ida.loni.usc.edu/pages/access/search.jsp')
    sleep(2)
    try:
        accept = driver.find_element(By.CSS_SELECTOR, '.ida-cookie-policy-right > div:nth-child(1)')
        accept.click()
    except NoSuchElementException:
        print("没有找到元素！")

    sleep(3)
    log = driver.find_element(By.CSS_SELECTOR, 'div.ida-menu-option:nth-child(4) > div:nth-child(1)')
    log.click()

    email_input = driver.find_element(By.CSS_SELECTOR, '.ida-menu-email-container > input:nth-child(2)')
    email_input.send_keys('21321306@hdu.edu.cn')

    password_input = driver.find_element(By.CSS_SELECTOR, '.ida-menu-password-container > input:nth-child(3)')
    password_input.send_keys('@CHh20022001CYFZYF')

    logging = driver.find_element(By.CSS_SELECTOR, '.login-btn > span:nth-child(2)')
    logging.click()

def add(driver,ID,group,bol):
    print('----------------------------------------------------------------------------------------------------------------------------')

    AdvSearch = driver.find_element(By.CSS_SELECTOR, '#advSearchTabId > a:nth-child(1) > em:nth-child(1) > font:nth-child(1)')
    AdvSearch.click()
    sleep(6)

    SubID = driver.find_element(By.CSS_SELECTOR, '#subjectIdText')
    SubID.clear()
    SubID.send_keys(ID)

    if bol == True:
        # 侧边栏选项
        Pre_p = driver.find_element(By.CSS_SELECTOR, '#preProcessedOption')
        Pre_p.click()
        Post_p = driver.find_element(By.CSS_SELECTOR, '#postProcessedOption')
        Post_p.click()
        Image_p = driver.find_element(By.CSS_SELECTOR, '#imageProcessingOption')
        Image_p.click()
        print('侧边栏勾选完成！')

        # Study栏勾选
        StudyDate = driver.find_element(By.CSS_SELECTOR, '#RESET_STUDY\.0')
        StudyDate.click()
        print('Study栏勾选完成！')

        # IMAGE栏勾选
        Display = driver.find_element(By.CSS_SELECTOR, '#RESET_MODALITY\\.0')
        Display.click()
        print('IMAGE栏勾选完成')

        # IMAGE PROTOCOL栏勾选
        ThreeD = driver.find_element(By.CSS_SELECTOR, '#imgProtocol_checkBox1\.Acquisition_Type\.3D')
        ThreeD.click()
        # FDG = driver.find_element(By.CSS_SELECTOR, '#imgProtocol_checkBox4\.Radiopharmaceutical\.18F-FDG')
        # FDG.click()
        print('IMAGE PROTOCOL栏勾选完成')

        # IMAGE PROCESSING栏勾选
        image_volume = driver.find_element(By.CSS_SELECTOR, '#fileType_checkBox1')
        image_volume.click()
        dis1 = driver.find_element(By.ID, 'RESET_PROCESSING.IMAGE_PROCESSING.ANATOMIC_STRUCTURE')
        dis1.click()
        dis2 = driver.find_element(By.ID, 'RESET_PROCESSING.IMAGE_PROCESSING.TISSUE_TYPE')
        dis2.click()
        dis3 = driver.find_element(By.ID, 'RESET_PROCESSING.IMAGE_PROCESSING.LATERALITY')
        dis3.click()
        dis4 = driver.find_element(By.ID, 'RESET_PROCESSING.IMAGE_PROCESSING.REGISTRATION')
        dis4.click()

        print('IMAGE PROCESSING栏勾选完成')

        return False

    try:
        SEARCH = driver.find_element(By.ID, 'advSearchQuery')
        SEARCH.click()
        sleep(3)

        element = driver.find_element(By.ID, 'advTableDescription')
        text = element.text
        words = text.split()  # 分割字符串
        last_word = words[-1]  # 获取最后一个单词
        number = int(last_word)  # 将字符串转化为数字
        times = number // 20 + 1

        for i in range(times):

            # 定位到table元素
            table = driver.find_element(By.CSS_SELECTOR, '#advTableData > table:nth-child(2)')

            # 获取所有的行
            rows = table.find_elements(By.TAG_NAME, 'tr')

            # 遍历每一行
            for row in rows:
                # 获取这一行的所有列
                cols = row.find_elements(By.TAG_NAME, 'td')

                # 检查第10列的值是否为MPRAGE
                if cols[9].text == 'MPRAGE' or cols[9].text == 'MP-RAGE':
                    # 点击第7列的input元素
                    cols[6].find_element(By.TAG_NAME, 'input').click()

            drag = driver.find_element(By.CSS_SELECTOR,
                                       '#advScrollBarTable > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(3) > td:nth-child(1) > input:nth-child(1)')
            if i != times - 1:
                drag.click()
                sleep(1)

        add2Coll = driver.find_element(By.CSS_SELECTOR, '#advResultAddCollectId')
        add2Coll.click()

        input_group = driver.find_element(By.CSS_SELECTOR, '#nameText')
        input_group.send_keys(group)
        sleep(1)

        buttons = driver.find_elements_by_tag_name('button')
        for button in buttons:
            if button.text == 'OK':
                button.click()
                sleep(5)
                break
        print(ID + '添加成功')
    except Exception as e:
        print(ID + '发现异常', e)

    print('----------------------------------------------------------------------------------------------------------------------------')




    # OK_button = driver.find_element(By.CSS_SELECTOR,'#yui-gen2 > span:nth-child(1) > button:nth-child(1)')
    # driver.execute_script("arguments[0].click();", OK_button)

start = 1
while True:

    try:
        last = 0

        driver = webdriver.Firefox()
        logging(driver)
        sleep(2)
        # 读取csv文件
        df = pd.read_csv('filt_2&5_3year.csv')

        # 选择第二行起第二列的数据
        data = df.iloc[start:, 1]

        # 将数据转换为数组
        data_array = data.values
        driver.get('https://ida.loni.usc.edu/pages/access/search.jsp?tab=advSearch&project=ADNI&page=DOWNLOADS&subPage=IMAGE_COLLECTIONS')
        bol = True
        group = '2&5_3year'
        preID = ''
        for i, ID in enumerate(data_array):
            last = i
            if preID != ID or i == 0:
                print(ID + '添加进组' + group)
                bol = add(driver,ID,group,bol)

            preID = ID
    except Exception as e:
        start = start + last
        print('在第' + str(start) + '次运行中程序因为' + str(e) + '终断，继续执行')
