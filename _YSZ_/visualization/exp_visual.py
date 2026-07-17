import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn import tree
import numpy as np

def visualization_maze_3d(EXP, z_max=-1):
    import matplotlib.pyplot as plt
    colors = ['red', 'blue', 'green', 'orange']
    EXP_visual = np.unique(EXP, axis=0)  # 去除重复元素
    # 逐个统计数量
    Z = np.zeros(len(EXP_visual), dtype=np.int32)
    for i in range(len(EXP_visual)):
        for j in range(len(EXP)):
            if (EXP_visual[i] == EXP[j]).all():
                Z[i] += 1
    # 打开画图窗口1，在三维空间中绘图
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    for i in range(len(EXP_visual)):
        # 给出点（0，0，0）和（100，200，300）
        x = [EXP_visual[i][0], EXP_visual[i][0]]
        y = [EXP_visual[i][1], EXP_visual[i][1]]
        z = [0, Z[i]]
        # 将数组中的前两个点进行连线
        ax.plot(x, y, z, linewidth=2, c=colors[np.int32(EXP_visual[i][2])])
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(-20, 20)
        if z_max != -1:
            ax.set_zlim(0, z_max)
        ax.set_xlabel('s[0]')
        ax.set_ylabel('s[1]')
        ax.set_zlabel('count')
    plt.show()

def data_visual_2d(X, Y):
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #markers = ["o", "*", "+"]
    col = []
    for i in range(0, len(Y)):
        col.append(colors[int(Y[i])])
    plt.scatter(X[:, 0], Y[:, 1], marker='o', c=col)
    plt.show()