import csv
import random
# from pandocfilters import Math
from scipy import stats
import torch
import numpy as np
# from config import args
# 类需要继承Function类，此处forward和backward都是静态方法
import gb_accelerate_temp as new_GBNR
# import gb2 as new_GBNR

"""
numbers : 每个球内部样本数
balls: [ball_sample_numbers,sample_label+64维样本向量]
result: ball_label+64维球中心向量
radius: 每个球半径
"""


def calculate_distances(center, p):
    return ((center - p) ** 2).sum(axis=0) ** 0.5


class GBNR(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        # print(input_,'----------------')
        # input_.shape :  [bs,68(pur+ori+index+now_label+64维向量)]
        self.batch_size = input_.size(0)
        input_main = input_[:, 1:]  # noise_label+64 [bs,65]
        self.input = input_[:, 2:]  # backward中使用
        pur = input_[:, 0].numpy().tolist()[0]  # 从第0维取出纯度
        #print("纯度为：", pur)

        self.flag = 0
        # print(input_main)
        numbers, balls, result, radius = new_GBNR.main(input_main, pur)  # 加了pur
        for re in range(3):
            # 用于存放合格球
            numbers1 = []
            balls1 = []
            center1 = []
            # 用于存放需再次聚类球
            input2 = []
            # 计数
            count = 0
            index = 0
            for i in balls:
                if len(i) <= 3 and count < input_.shape[0]:  # 最大设为batchsize，将所有单样本求进行再次聚类args.BATCH_SIZE:
                    for plot in i:
                        input2.append(plot)
                        count += len(i)
                else:
                    balls1.append(i)
                    numbers1.append(numbers[index])
                    center1.append(result[index])
                index += 1

            if index < len(numbers):  # 到达当前index时count先满足条件，则后续球自动归为合格类
                while index < len(numbers):
                    balls1.append(balls[index])
                    numbers1.append(numbers[index])
                    center1.append(result[index])
                    index += 1

            if len(input2) != 0:  # 用于二次聚类球
                input2 = torch.Tensor(input2)
                numbers3, balls3, result3, radius3 = new_GBNR.main(input2, pur)
                index1 = 0
                while index1 < len(numbers3):
                    balls1.append(balls3[index1])
                    numbers1.append(numbers3[index1])
                    center1.append(result3[index1])
                    index1 += 1

            balls = balls1  # 赋值
            numbers = numbers1  #
            result = center1  #
            pur -= 0.025  # 每聚一次pur-0.05

        numbers4 = []
        balls4 = []
        center4 = []
        for i in range(len(numbers1)):
            if len(balls1[i]) >= 1:
                balls4.append(balls1[i])
                center4.append(center1[i])
                numbers4.append(numbers1[i])
        if(len(numbers4)==0):
            balls4=balls1
            center4=center1
            numbers4=balls1
        for i in range(1, len(numbers4)):
            for j in range(len(numbers4)-i):
                if numbers4[j] < numbers4[j+1]:
                    numbers4[j], numbers4[j+1] = numbers4[j+1], numbers4[j]
                    balls4[j], balls4[j+1] = balls4[j+1], balls4[j]
                    center4[j], center4[j+1] = center4[j+1], center4[j]

        self.balls = balls4  # 各球内样本
        self.numbers = numbers4  # 各球内样本数
        self.center = center4  # 求中心
        target = []
        data = []
        count = 0
        
        for k, i in enumerate(center4):  # i:[ball_label, 64维球心]

            if k < int(1/2 * len(numbers4)):  # 1/2球的样本
                if len(balls4[k])==1 and int(i[0])==0:  # 
                  continue
                data.append(i[1:])  # 球中心向量
                target.extend(np.array([i[0]]))  # ball—label
                
                print('求内样本总数:',len(balls4[k]))
                print('求内样本标签___:',int(i[0]))
                
                if len(balls4[k])==1:
                    count+=1
        self.data = np.array(data)  # [ball_number, 64维球心]
        print('总样本数量',len(data))
        print('单样本个数',count)
        return torch.Tensor(data), torch.Tensor(target),count
    @staticmethod
    def backward(self, output_grad, tar,sb):
        # forward 输出参数数量必须和 backward输入一样

        balls = np.array(self.balls)
        result = np.zeros([self.batch_size, 152], dtype='float64')

        for i in range(output_grad.size(0)):

            # 这是将中心梯度返回给球里面的每一个样本
            for a in balls[i]:
              result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 2:] = np.array(output_grad[i, :])
            
            # 这是将中心梯度传给球里面的1/3的样本
            # 如果是第0类的话就不完全反向传播梯度，而是随机取1/5的点传播
            # if int(self.center[i][0])==0:
            #     for a in random.sample(balls[i].tolist(),max(balls[i].shape[0]//5,1)):
            #         a = balls[i][random.randint(0,balls[i].shape[0]-1)]
            #         result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 2:] = np.array(output_grad[i, :])
            # else:
            #     for a in balls[i]:
            #         result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 2:] = np.array(output_grad[i, :])


            
              
        return torch.Tensor(result)
