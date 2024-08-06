import torch
import torch.nn as nn


"""
瓶颈层。虽然每层只产生k个输出特征映射，它通常有更多的输入。
在[37,11]中已经注意到1×1卷积可以在每次3×3卷积之前引入瓶颈层，
以减少输入特征映射数量，从而提高计算效率。
"""


#实现了 DenseNet 中的瓶颈层，瓶颈层包含一个 1x1 的卷积层，后接一个 3x3 的卷积层。
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # """In  our experiments, we let each 1×1 convolution
        # produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        """我们发现这种设计对DenseNet和表示我们的网络有这样一个瓶颈层，
        即到BN-ReLU-Conv(1×1)-BN-ReLUConv(3×3)的H版本，作为DenseNet-B。"""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),  # 批量归一化
            nn.ReLU(inplace=True),  # RELU激活
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),  #1x1卷积
            nn.BatchNorm2d(inner_channel),  # 批量归一化
            nn.ReLU(inplace=True),  # RELU激活
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)  # 3x3卷积
        )

    # 特征链接
    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


"""
块之间的层称为过渡层，做卷积和池化。
"""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):  # 初始化
        super().__init__()
        
        # 过渡层由批处理规范化层和1×1卷积层，然后接一个2×2平均池化层
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

# DesneNet-BC
# B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
# C stands for compression factor(0<=theta<=1)  压缩因子？


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=200):
        super().__init__()
        self.growth_rate = growth_rate

        # 在进入第一个密集块之前，进行一个16(或DenseNet-BC增长率的两倍，本实验为32*2=64)输出通道的在输入图像上执行的卷积
        inner_channels = 2 * growth_rate

        # #对于内核大小为3×3的卷积层，每个输入的边是零填充一个像素来保持特征映射大小固定。
        #For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel to keepthe feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        # 特征提取部分
        self.features = nn.Sequential()

        # 遍历每个密集块，
        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))  # 使用 _make_dense_layers 方法创建密集块的层，并添加到 self.features中。
            inner_channels += growth_rate * nblocks[index]  # 更新 inner_channels，增加特征图数量。

            # 如果一个密集块包含m个特征映射，我们让下面的过渡层产生 θ * m 个输出特征图，其中0 < θ ≤ 1称为压缩因子。
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        
        # 添加最后一个密集块，添加批量归一化层（BatchNorm2d）和激活函数（ReLU）。
        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        
        # 全局平均池层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)  # 卷积
        output = self.features(output)  # 密集块
        output = self.avgpool(output)  # 全局平均池化
        output = output.view(output.size()[0], -1)  # 展平
        output = self.linear(output)  # 全连接
        return output

    # 创建密集块
    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):  # nblocks个密集块
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)  # []对应第几组密集块有几层，增长率已设定为32

