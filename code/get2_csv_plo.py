import csv
import os

import pandas as pd
import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
add2 = "./data_set/data2_extract/"

def data_draw (f):
    # lista=[i for i in range(200)]
    for n in range(50):
        data_2 = pd.read_csv(add2 + f + "/" + f + "-" + str(n + 1) + ".csv")  # 读取csv文件,names=['dCh1','dCh2','dCh3','dCh4']
        data_2 = np.array(data_2)
        # print(data2_extract)
        # print("./data_set/data2_extract/" + f + "/" + f + "-" + str(n + 1) + ".csv")
        # x=range(100)
        # plt.ylim(0,0.4)
        line1,=plt.plot(data_2[1],'r--',linewidth=0.8) #data2_extract[0],
        line2,=plt.plot(data_2[2],'b-',linewidth=0.8)
        line3,=plt.plot(data_2[3],'g-',linewidth=0.8)
        line4,=plt.plot(data_2[4],'y-',linewidth=0.8)
        ll=plt.legend([line4,line3,line2,line1],["Ch4","Ch3","Ch2","Ch1"],loc='upper right')
        '''
        ax2 = plt.gca()
        ax2.spines['top'].set_visible(False)  #去掉上边框
        ax2.spines['right'].set_visible(False) #去掉右边框
        '''
        plt.grid(axis="y",linestyle='--')            #b, which, axis, color, linestyle, linewidth， **kwargs
        # plt.text(1,1,'Ch1 ',fontdict={'size': 9, 'color':  'blue'})       #字体尺寸9，颜色 蓝色   第一和第二个参数60000,0.13表示输出信息的坐标，原点坐标是（0，0）
        # plt.text(1,2,'Ch2 ',fontdict={'size': 9, 'color':  'red'})
        plt.ylabel("C(PF)",fontsize=11)     #设置纵轴单位
        plt.xlabel("step",fontsize=11)         #设置横轴单位
        #plt.title(" ",fontsize=11)            #设置图片的头部
        path='./data_set/pic_2/'+f
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig("./data_set/pic_2/"+f+"/"+f+"-"+str(n+1)+".png",dpi=500)       #图片保存位置，图片像素
        plt.rcParams['figure.dpi'] =900        #分辨率
        # plt.show()
        plt.clf()
        #存入数据集


if __name__ == '__main__':
    data_draw ("CA")
    data_draw ("HD")
    data_draw ("QP")
    data_draw ("ZD")
