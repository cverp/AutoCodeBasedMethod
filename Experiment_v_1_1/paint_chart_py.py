# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
import scipy.io as scio
mpl.rcParams['font.sans-serif'] = ['SimHei']

def paint_chart(y,y_name):
    y_len=len(y)
    names=[xc for xc in range(0, y_len)]
    x = range(len(names))
    plt.plot(x, y,  mec='r', mfc='w')
    plt.legend()  # 让图例生效
    #plt.xticks(x, names, rotation=45)
    #plt.margins(0)
    #plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"time(s)")  # X轴标签
    plt.ylabel(y_name)  # Y轴标签
    plt.title(y_name)  # 标题
    plt.show()
def paint_chart1(y,y_name):
    y_len=len(y)
    names=[xc for xc in range(0, y_len)]
    x = range(len(names))
    plt.plot(x, y,  mec='r', mfc='w')
    plt.legend()  # 让图例生效
    #plt.xticks(x, names, rotation=45)
    #plt.margins(0)
    #plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"time(s)")  # X轴标签
    plt.ylabel(y_name)  # Y轴标签
    plt.title(y_name)  # 标题
    plt.show()
'''
def paint_chart_after(mat_file,mat_name,y_name):
    data = scipy.io.loadmat(mat_file)  # 读取mat文件
    y = np.array(data[mat_name])
    y_len = len(y)
    names = [xc for xc in range(0, y_len)]
    x = range(len(names))
    plt.plot(x, y, mec='r', mfc='w')
    plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"time(s)")  # X轴标签
    plt.ylabel(y_name)  # Y轴标签
    plt.title(y_name)  # 标题
    plt.show()
    #print(x)
    return x
    
'''
if __name__ == '__main__':
    matfile=""
    matname=""
    '''
    paint_chart_after(mat_file, mat_name, "||...||+alpha||...||+beta||...||")
    '''

    '''
names = ['5', '10', '15', '20', '25']
x = range(len(names))
y = [0.855 ]
y.append(0.855)
y.append(0.155)
y.append(0.855)
y.append(0.155)
y1=[0.86,0.85,0.853,0.849,0.83]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'y=x^2曲线图')
plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3曲线图')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"time(s)邻居") #X轴标签
plt.ylabel("RMSE") #Y轴标签
plt.title("A simple plot") #标题

plt.show()
    
    '''