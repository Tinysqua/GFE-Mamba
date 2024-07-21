import os

def Makedir(path):
    folder = os.path.exists(path)
    if (not folder):
        os.makedirs(path)

def GetFileName(fileDir, outDir):
    list_name = []
    Makedir(outDir)
    for dir in os.listdir(fileDir):
        filePath = os.path.join(fileDir, dir)
        print(filePath)

        txt = os.path.join(outDir, "donelist.txt")

        f = open(txt, 'a')
        f.write(filePath+'\n')
        f.close()

def main():
    print("procesiing……\n")
    fileDir = r"E:\PythonProject\MRIGet\Find\find_\ADNI_1"  # 输入文件夹路径
    outDir = "process_output"
    files = GetFileName(fileDir, outDir)
    print("done!\n")

if __name__ == "__main__":
    main()
