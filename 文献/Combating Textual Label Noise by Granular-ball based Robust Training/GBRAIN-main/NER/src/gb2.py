import math
import time
import pandas as pd

import numpy as np
import numpy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

# 1.输入数据data
# 2.打印绘制原始数据
# 3.判断粒球的纯度
# 4.纯度不满足要求，k-means划分粒球
# 5.绘制每个粒球的数据点
# 6.计算粒球均值，得到粒球中心和半径，绘制粒球


# 判断粒球的标签和纯度
def get_label_and_purity(gb):
    # 分离不同标签数据
    len_label = numpy.unique(gb[:, 0], axis=0)
    # print("len_label\n", len_label)  # 球内所有样本标签（不含重复)

    if len(len_label) == 1:  # 若球内只有一类标签样本，则纯度为1，将该标签定为球的标签
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 0] == label)] = label
        # print("分离\n", gb_label_temp)  # dic{该标签对应样本数：标签类别}
        # 粒球中最多的一类数据占整个的比例
        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0  # pur为球内同一标签对应最多样本的类的样本数/球内总样本数

        label = gb_label_temp[max_label]  # 对应样本最多的一类定为球标签
    # print(label)
    # 标签、纯度
    return label, purity


# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]
    # print(data_no_label)
    center = data_no_label.mean(axis=0)
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius


def splits(gb_list, purity, splitting_method):
    gb_list_new = []
    for gb in gb_list:
        label, p = get_label_and_purity(gb)

        if int(label)==0:
            p1 = 1
        elif int(label)==3:
            p1 = 1
        else:
            p1=purity
        if p >= p1 and len(gb)<=50:
            gb_list_new.append(gb)
        else:
            gb_list_new.extend(splits_ball(gb, splitting_method))
    return gb_list_new


# 距离
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5


def splits_ball(gb, splitting_method):
    splits_k = 4
    data_no_label = gb[:, 1:]
    ball_list = []
    label = []
    # print('splits_k', splits_k)
    # 数组去重
    len_no_label = numpy.unique(data_no_label, axis=0)

    if splitting_method == '2-means':
        if len(len_no_label) < splits_k:
            splits_k = len(len_no_label)
        # X: 数据; n_clusters: K的值; random_state: 随机状态（为了保证程序每次运行都分割一样的训练集和测试集）
        label = k_means(X=data_no_label, n_clusters=splits_k, n_init=1, random_state=5)[1]  # 返回标签
    elif splitting_method == 'center_split':
        # 采用正、负类中心直接划分
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        distances_to_p_left = distances(data_no_label, p_left)
        distances_to_p_right = distances(data_no_label, p_right)

        relative_distances = distances_to_p_left - distances_to_p_right
        label = numpy.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))

    elif splitting_method == 'center_means':
        # 采用正负类中心作为 2-means 的初始中心点
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        centers = numpy.vstack([p_left, p_right])
        label = k_means(X=data_no_label, n_clusters=2, init=centers, n_init=10)[1]
    else:
        return gb
    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])
    # print(ball_list)
    return ball_list


def main(data,pur):


    # 数组去重
    data = numpy.unique(data, axis=0)
    purity = pur

    # 直接绘制输入数据
    gb_list = [data]
    # print(len(data))
    # gb_plot(gb_list)

    while True:
        ball_number_1 = len(gb_list)
        gb_list = splits(gb_list, purity=purity, splitting_method='2-means')
        ball_number_2 = len(gb_list)
        # gb_plot(gb_list)
        if ball_number_1 == ball_number_2:  # 粒球数和上一次划分的粒球数一样，即不再变化
            break
    # 绘制纯度符合要求的粒球
    # gb_plot(gb_list, 1)
    # 绘制纯度符合要求的粒球的中心
    # gb_plot(gb_list, 2)



    centers = []
    numbers = []
    radius = []
    # print('总平均耗时：%s' % (round(times / 3 * 1000, 0)))
    index = []
    result = []
    for i in gb_list:  # 遍历每个球
        a = list(calculate_center_and_radius(i)[0])
        radius1 = calculate_center_and_radius(i)[1]  # a 每个半径
        lab, p = get_label_and_purity(i)  # 获取每个球标签、纯度
        a.insert(0, lab)  # 下标为0的位置（首位）插入球标签
        # print("a+label\n", a)  # [1,1(label)+64]
        centers.append(a)
        radius.append(radius1)
        result.append(i)
        # print("result:\n", gb_dict[i][0])  # [1,1(ball_label)+64]
        index1 = []
        for j in i:
            index1.append(j[-1])
            # print("index1\n", index1)  # 球心64维向量最后一维
        numbers.append(len(i))  # 球内部样本数
        # print("ball_samples:\n", len(gb_dict[i][-1]))
        index.append(index1)
    # print("centers\n", centers)
    # print("results\n", result)

    return numbers, result, centers, radius






if __name__ == '__main__':
    data = np.array([
        [1, 0.1, 0.5],
        [1, 0.31, 0.5],
        [1, 0.32, 0.5],
        [1, 0.22, 0.51],
        [0,-0.5,-0.5]
    ])
    numbers, result, centers, radius = main(data,1)

    print(numbers)
    print(result)
    print(centers)
    print(radius)
