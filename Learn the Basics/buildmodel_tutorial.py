"""
`学习基础知识 <intro.html>`_ ||
`快速开始 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集和数据加载器 <data_tutorial.html>`_ ||
`变换  <transforms_tutorial.html>`_ ||
**创建模型** ||
`自动求导 <autogradqs_tutorial.html>`_ ||
`优化 <optimization_tutorial.html>`_ ||
`保存和加载模型 <saveloadrun_tutorial.html>`_

构建神经网络
===================

神经网络由对数据进行操作的层/模块组成。
 `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ 命名空间提供了您构建自己的神经网络所需的所有构建模块。 PyTorch中的每个模块都继承了  `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.
一个神经网络本身就是一个由其他模块（层）组成的模块。这种嵌套结构允许轻松构建和管理复杂的架构。

在下面的章节中，我们将建立一个神经网络来对FashionMNIST数据集中的图像进行分类。

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# 获取训练设备
# -----------------------
# 我们希望能够在GPU或MPS等硬件加速器上训练我们的模型。
# 如果可以使用的话，让我们检查一下 `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_
# 或者 `torch.backends.mps <https://pytorch.org/docs/stable/notes/mps.html>`_ 是否是可用的，否则我们使用CPU。

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

##############################################
# 定义网络结构类
# -------------------------
# 我们通过继承 ``nn.Module``来创建神经网络, 并且使用 ``__init__``方法
# 初始化神经网络层。 每个 ``nn.Module`` 都在 ``forward`` 方法中实现对输入数据的操作。

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

##############################################
# 我们创建一个 ``NeuralNetwork`` 的实例, 然后将其移动到 ``device`` 上， 并打印
# 其结构。

model = NeuralNetwork().to(device)
print(model)


##############################################
# 为了使用这个模型，我们把输入数据传给它。这就执行了模型的 ``forward``方法,
# 以及一些 `background operations <https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866>`_.
# 不要直接调用 ``model.forward()`` ！
#
# 在这个输入上调用模型会返回一个二维张量，维度为0的数值对应于每个类的10个原始预测值的每个输出，维度为1的数值对应于每个输出的单个数值。
# 我们通过``nn.Softmax`` 模块的一个实例来获得预测概率。

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


######################################################################
# --------------
#


##############################################
# 模型层
# -------------------------
#
# 让我们来分解FashionMNIST模型中的各个层次。为了说明它，我们
# 将采取一个由3张大小为28x28的图像组成的样本小批量把它通过网络时看看发生了什么。

input_image = torch.rand(3,28,28)
print(input_image.size())

##################################################
# nn.Flatten
# ^^^^^^^^^^^^^^^^^^^^^^
# 我们初始化 `nn.Flatten  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_
# 层去转换每个2D的28*28的图片变成一个包含784个像素值的连续数组  (
# 小批量维度 (维度=0)保持原来的数值).

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

##############################################
# nn.Linear
# ^^^^^^^^^^^^^^^^^^^^^^
# `linear layer <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
# 是一个使用其存储的权重和偏置，对输入进行线性转换的模块。
#
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


#################################################
# nn.ReLU
# ^^^^^^^^^^^^^^^^^^^^^^
# 非线性激活是在模型的输入和输出之间建立复杂的映射关系。
# 它们被应用在线性变换之后，以引入非线性，帮助神经网络
# 学习各种各样的现象。
#
# 在这个模型中, 我们使用 `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ 在我们的
# 线性层，但还有其他的激活方式可以在你的模型中引入非线性。

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")



#################################################
# nn.Sequential
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ 是一个包含多个模块的有序容器
# 数据通过所有的模块按照定义好的顺序。你可以使用
# 序列容器来拼凑一个快速的网络，例如 ``seq_modules``.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

################################################################
# nn.Softmax
# ^^^^^^^^^^^^^^^^^^^^^^
# 神经网络的最后一个线性层返回 `logits` - 原始数值在 [-\负无穷, \无穷] 里- 这将传递给
# `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ 模块。结果被缩放到数值
# [0, 1] 之间表示模型对于每个类的预测概率。 ``dim`` 这个参数表示维度的数值必须和为1。

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#################################################
# 模型参数
# -------------------------
# 神经网络内部的许多层都是参数化的，即有相关的权重和偏差，在训练过程中被优化，
# 自动继承 ``nn.Module`` 类。
# 追踪你的模型对象中定义的所有字段，并使所有的参数
# 是可获得的 ，使用模型的 ``parameters()``方法或者 ``named_parameters()`` 方法。
#
# 在这个例子中， 我们遍历每个参数，并打印其大小和预览其值。
#


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

######################################################################
# --------------
#

#################################################################
# 更多阅读
# --------------
# - `torch.nn API <https://pytorch.org/docs/stable/nn.html>`_
