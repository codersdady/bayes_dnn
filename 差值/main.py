import numpy as np
from scipy.spatial.distance import pdist
import csv


x=input()

with open(x, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
    while not x == "!":
        num=0
        x=input()
        a = float(input("请输入a"))
        b = float(input("请输入b"))
        # c = float(input("请输入c"))
        # d = float(input("请输入d"))
        aa = float(input("请输入aa"))
        bb = float(input("请输入bb"))
        # cc = float(input("请输入cc"))
        # dd = float(input("请输入dd"))
        num+=pow(a-aa,2)
        num+=pow(b-bb,2)
        # num+=pow(c-cc,2)
        # num+=pow(d-dd,2)
        num= np.math.sqrt(num)
        f.write(x+": "+str(num))
        f.write("\n")
        x=input()
f.close()