import csv
from config import opt
# from pandocfilters import Math
from scipy import stats
import torch
import numpy as np
# from config import args
# 类需要继承Function类，此处forward和backward都是静态方法
import gb_accelerate_temp as new_GBNR


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

        # input_.shape :  [bs,68(pur+ori+index+now_label+64维向量)]
        self.batch_size = input_.size(0)
        input_main = input_[:, 3:]  # noise_label+64 [bs,65]
        self.input = input_[:, 4:]  # backward中使用
        self.res = input_[:, 1:2]  # 样本原标签
        # print("self.res: ", self.res)
        self.index = input_[:, 2:3]  # .numpy().tolist()
        # print("self.index: ", self.index)
        """
        self.index:  tensor([[31819.],
        [19405.],
        [37096.],
        ...,
        [20207.],
        [49052.],
        [43082.]], requires_grad=True)
        """
        pur = input_[:, 0].numpy().tolist()[0]  # 从第0维取出纯度
        #print("纯度为：", pur)

        self.flag = 0

        numbers, balls, result, radius = new_GBNR.main(input_main, pur)  # 加了pur


        # for re in range(args.recluster):
        for re in range(opt.recluster):
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
                if len(i) < opt.min_ball and count < opt.batch_size:  # 最大设为batchsize，将所有单样本求进行再次聚类args.BATCH_SIZE:
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
            pur -= 0.05  # 每聚一次pur-0.05

        numbers4 = []
        balls4 = []
        center4 = []
        for i in range(len(numbers1)):
            if len(balls1[i]) >= opt.min_ball:  # 取出数量大于最小值的球
                balls4.append(balls1[i])
                center4.append(center1[i])
                numbers4.append(numbers1[i])
        if(len(numbers4)==0):
            balls4=balls1
            center4=center1
            numbers4=balls1
        #print(opt.recluster, "次聚类去除小样本后共有：", len(numbers4), "个球")
        # print(args.recluster, "次聚类去除小样本后共有：", len(numbers4), "个球")
            

        
            

        # 依据各球内样本数对number、center、balls进行重新排序(冒泡排序): 从大到小
        # print("number4--before:", numbers4)
        for i in range(1, len(numbers4)):
            for j in range(len(numbers4)-i):
                if numbers4[j] < numbers4[j+1]:
                    numbers4[j], numbers4[j+1] = numbers4[j+1], numbers4[j]
                    balls4[j], balls4[j+1] = balls4[j+1], balls4[j]
                    center4[j], center4[j+1] = center4[j+1], center4[j]

        if False:
            n_numbers4 = []
            n_balls4 = []
            n_center4 = []
            slice_cnt = 1
            chunK = 1
            
            for i in range(len(numbers4)):
                gbcnt = numbers4[i]
                # if numbers4[i] <=opt.chunk_ball:
                #     continue
                if gbcnt <= slice_cnt:
                    n_numbers4.append(gbcnt)
                    n_balls4.append(balls4[i])
                    n_center4.append(center4[i])
                else:
                    for j in range(0,gbcnt,chunK):
                        chunk_balls = balls4[i][j:j+chunK]
                        chunk_center = chunk_balls.mean(axis=0)
                        chunk_center[0] = center4[i][0]
                        n_numbers4.append(chunk_balls.shape[0])
                        n_balls4.append(chunk_balls)
                        n_center4.append(chunk_center)
            if len(n_numbers4) >0:
                numbers4 = n_numbers4
                balls4 = n_balls4
                center4 = n_center4



        self.balls = balls4  # 各球内样本
        self.numbers = numbers4  # 各球内样本数

       

        sample_index = []  # 保存前*%球内部的样本
        index_chaifen_id = []
        # expectation_index = []
        target = []
        data = []
        oneFlg = 0
        for k, i in enumerate(center4):  # i:[ball_label, 64维球心]
            p_b = opt.drop_ball*len(numbers4)
            if oneFlg ==0 and numbers4[k]==1:
                oneFlg = k
            if k < int(p_b) or int(p_b) <=1:# or numbers4[k] >1 or k<oneFlg+1:  # 大于1 或者在前p_b%
                data.append(i[1:])  # 球中心向量，
                target.extend(np.array([i[0]]))  # 球心标签
                count_temp=0
                count_temp2=0
                count_temp3=0

                label,p = new_GBNR.get_label_and_purity(balls4[k])
                # 达到原始纯度的粒球
                if p>=opt.purity or True:
                    for a in balls4[k]:
                        idx = self.index[np.where((np.array(self.input) == a[1:]).all(axis=1))][0]
                        or_label = self.res[np.where((np.array(self.input) == a[1:]).all(axis=1))][0]
                        center_label = center4[k][0]
                        sample_index.append((int(idx.item()),int(center_label)))

                        if or_label!=a[0]:
                            count_temp3+=1

                        if a[0]==center_label:


                            count_temp+=1
                        else:
                            count_temp2+=1
                    # print("标签和球心相同数量：",count_temp)
                    # print("标签和球心不同数量：",count_temp2)
                    # print("一个球中噪声标签数量：",count_temp3)
                    index_chaifen_id.append(len(balls4[k]))
              
        
        self.data = np.array(data)  
        #  求每个球中的向量
        gbid = []
        for gbs in balls4:
            buffId = []
           
            
            for v in gbs:
                idv = np.where((np.array(self.input) == v[1:]).all(axis=1))
                try:
                    buffId.append(idv[0].item())
                except:
                    print(idv)
               
               
            gbid.append(buffId)
        

        return torch.Tensor(data), torch.Tensor(target), torch.tensor(sample_index), torch.tensor(index_chaifen_id),gbid
  

    @staticmethod
    def backward(self, output_grad, input, index, id,_):
        balls = np.array(self.balls)
        result = np.zeros([self.batch_size, 154], dtype='float64')  # +4

        for i in range(output_grad.size(0)):
            for a in balls[i]:
              result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 4:] = np.array(output_grad[i, :])
        return torch.Tensor(result)


