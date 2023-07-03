"""
**简介** ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动求导 <autogradyt_tutorial.html>`_ ||
`创建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
`理解模型 <captumyt.html>`_

PyTorch的简介
=======================

请跟随下面的视频学习，或者你可以选择在 `youtube <https://www.youtube.com/watch?v=IC0_FRiX-sw>`__上观看此视频。

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/IC0_FRiX-sw" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

PyTorch张量
---------------

教程从视频的 `03:50 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=230s>`__开始。

首先，我们导入pytorch包。

"""

import torch

######################################################################
# 张量（Tensor）的基本的操作。
# 首先，我们使用不同的方式创建张量：
# 

z = torch.zeros(5, 3)
print(z)
print(z.dtype)


#########################################################################
# 上面的代码创建了一个5x3的全 0 矩阵，并查询其数据类型，发现这些 0 是32位浮点数（float32)类型，
# 这也是PyTorch默认的类型。
# 
# 如果你想将此张量的数据类型换成整数类型应该怎么办呢？
# 你可以重写默认类型：

i = torch.ones((5, 3), dtype=torch.int16)
print(i)


######################################################################
# 你可以看到，当我们改变默认数据类型后，打印张量时输出了相应的数据类型。
# 
# 随机初始化学习权重是很常见的，为了结果的可重复性，通常会用一个特定的PRNG种子：
# 

torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print(" 一个随机张量r1：")
print(r1)

r2 = torch.rand(2, 2)
print("\n 一个与r1不同的随机张量r2：")
print(r2) # 与r1具有不同的随机值

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print("\n 与r1相同的随机张量r3：")
print(r3) # 与r1的值相同，应为我们使用了相同的随机数种子


#######################################################################
# PyTorch中的张量能直观地进行算术运算。相似形状的张量可以进行加法、乘法等操作。
# 对标量的操作分布在张量上：

ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # 对每个元素乘2
print(twos)

threes = ones + twos       # 由于ones和twos的形状相同，所以可以进行相加
print(threes)              # 张量按逐元素相加
print(threes.shape)        # 输出张量维度与输入张量维度相同

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# 运行注释掉的代码会报runtime error
# r3 = r1 + r2


######################################################################
# 这里是一些可用的算术运算的例子：
# 

r = (torch.rand(2, 2) - 0.5) * 2 # 值介于 -1和 1 之间
print(" 一个随机矩阵 r：")
print(r)

# 支持常用的算数运算：
print("\n r的绝对值：")
print(torch.abs(r))

# 三角函数“
print("\n r的反sin函数：")
print(torch.asin(r))

# 线性代数运算，如行列式和奇异值分解：
print("\n r的行列式的值：")
print(torch.det(r))
print("\n r的奇异值分解：")
print(torch.svd(r))

# 统计和集合运算：
print("\n r的平均值和方差：")
print(torch.std_mean(r))
print("\n r的最大值：")
print(torch.max(r))


##########################################################################
# 关于PyTorch张量，我们还有很多知识需要去学习。
# 这些知识包括如何在GPU上设置张量进行并行计算，关于这一点我们将在另一个视频中进行更深入的探讨。
# 
# PyTorch模型
# --------------
#
# 教程从视频的 `10:00 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=600s>`__开始。
#
# 如何在PyTorch中定义模型
#

import torch
import torch.nn as nn            # 为了使用 torch.nn.Module, 这是PyTorch中所有模型的父类
import torch.nn.functional as F  # 为了使用激活函数


#########################################################################
# .. figure:: /_static/img/mnist.png
#    :alt: le-net-5 diagram
#
# *Figure: LeNet-5*
# 
# 上面是LeNet-5的示意图，它是最早的卷积神经网络之一，也是深度学习爆发的驱动力之一。
# 它可以读取手写数字的小图像（MNIST数据集），并正确分类图像中代表的数字。
# 
# 以下是其工作原理的简略版本：
# 
# -  C1层是一个卷积层，它可以扫描输入的图像，并在训练期间学到图像的特征。
#    它的输出是一张特征图，表示它在图像中看到的每一个所学特征。这个 "激活图 "在第S2层中被下采样处理。
# -  C3层是另一个卷积层，这次是扫描C1的激活图以寻找特征之间的联系。
#    它还提出了一个描述这些特征组合的空间位置的激活图，该激活图在第4层被下采样。
# -  最后的全连接层，F5、F6和OUTPUT，是一个分类器，将最终的激活图，分类到代表10个数字的10个类中。
# 
# 我们如何使用代码表示这个简单的神经网络呢？
# 

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1个输入图像通道（黑色和白色），6个输出通道，5x5正方形卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 一个仿射（线性）运算： y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5是图像尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在一个(2, 2)窗口上的最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果尺寸是一个平方数，你只能指定一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x)) # relu是一个激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # size拿到了除批量维度（batch)外x的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


############################################################################
# 翻看这段代码，你应该能够发现与上图的一些结构上的相似之处。
# 
# 它展示了一个典型的PyTorch模型的结构：
#
# -  这个模型继承自 ``torch.nn.Module``  - 模块可以嵌套使用 - 事实上，
#     ``Conv2d`` 和 ``Linear`` 层类也继承于 ``torch.nn.Module``。
# -  一个模型会有一个实例化它用到的层的 ``__init__()`` 函数，
#    并在这里加载它可能需要的任何数据处理工具（例如，一个NLP模型可能加载一个词汇表）。
# -  一个模型会有一个 ``forward()`` 函数。
#    这就是实际计算发生的地方： 模型输入通过网络层和各种函数来生成一个输出。
# -  除此之外，你可以像其他的 Python 类一样建立你的模型类，
#    添加你需要的任何属性和方法来支持你的模型的计算。
# 
# 让我们实例化这个对象并通过一个样本输入运行它。
# 

net = LeNet()
print(net)                         # 打印网络结构

input = torch.rand(1, 1, 32, 32)   # 生成一个随机张量表示一张32x32的黑白图像
print('\n图像批量形状：')
print(input.shape)

output = net(input)                # 我们不直接调用forward()
print('\nRaw output:')
print(output)
print(output.shape)


##########################################################################
# 上面发生了几件重要的事情：
# 
# 首先，我们实例化了 ``LeNet`` 类， 并打印了 ``net`` 对象。
# ``torch.nn.Module`` 的子类将报告它所创建的层以及它们的形状和参数。
# 如果你想了解一个模型的处理要领，这提供了一个的概述。
# 
# 接下来，我们创建一个假的输入，代表一个32x32的单通道的图像。通常情况下，
# 你会加载一个图像并将其转换为这种形状的张量。
# 
# 你可能已经发现我们的张量有一个额外的维度 - *批处理维度。*
# PyTorch模型假定它们是在成批的数据上工作的--例如，一批16张的图像的形状是
# ``(16, 1, 32, 32)``。由于我们只使用一张图片，所以我们创建了一个形状为 ``(1, 1, 32, 32)``的批次。
# 
# 我们像调用函数一样调用模型来让它进行推理：
# ``net(input)``。 这个调用的输出代表了模型对输入代表一个特定数字的置信度。
# (由于该模型的这个实例还没有学到任何东西，我们不期望在输出中看到任何有用的推理）。
# 看一下输出的 ``输出``， 我们可以看到它也有一个批次维度，
# 其大小应该总是与输入批次维度相匹配。如果我们传入一个16个实例的输入批，
#  ``输出`` 的形状是 ``(16, 10)``。
# 
# 数据集和数据加载器
# ------------------------
#
# 教程从视频的 `14:00 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=840s>`__开始。
#
# 下面，我们将演示如何使用TorchVision的开放数据集，如何转换数据集中的图像供你的模型使用，
# 以及如何使用DataLoader向你的模型输入成批的数据。
#
# T我们需要做的第一件事是将我们传入的图像转换为PyTorch张量。
#

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


##########################################################################
# 这里，我们为输出指定了两种变换：
#
# -  ``transforms.ToTensor()`` 将Pillow加载的图片转换为了PyTorch张量。
# -  ``transforms.Normalize()`` 调整张量的值，使其平均值为零，标准差为0.5。
#      大多数激活函数在 x=0 附近有最大的梯度，因此将我们的数据集中在那里可以加速学习。
# 传递给转换的值是数据集中图像的RGB值的平均值（第一个元组）和标准差（第二个元组）。
# 你可以通过运行以下代码自己计算这些值几行代码来计算：
#          ```
#           from torch.utils.data import ConcatDataset
#           transform = transforms.Compose([transforms.ToTensor()])
#           trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
#
#           #stack all train images together into a tensor of shape 
#           #(50000, 3, 32, 32)
#           x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])
#           
#           #get the mean of each channel            
#           mean = torch.mean(x, dim=(0,2,3)) #tensor([0.4914, 0.4822, 0.4465])
#           std = torch.std(x, dim=(0,2,3)) #tensor([0.2470, 0.2435, 0.2616])  
# 
#          ```   
# 
# 还有许多可用的变换，包括裁剪、居中、旋转和反射。
# 
# 接下来，我们将创建一个CIFAR10数据集的实例。这是一组32x32的彩色图像，
# 包含10类物体： 6种动物（鸟、猫、鹿、狗、青蛙、马）和4种交通工具（飞机、汽车、船、卡车）：
# 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


##########################################################################
# .. 小贴士::
#      当你运行上面的代码块时，可能会花一些时间下载数据集。
# 
# 这是一个在PyTorch中创建数据集对象的例子。下载的数据集（如上面的CIFAR-10）是
# ``torch.utils.data.Dataset``的子类。 PyTorch中的``数据集``
# 包括TorchVision、Torchtext和TorchAudio中的可下载数据集，以及实用数据集类，
# 如 ``torchvision.datasets.ImageFolder``,它将读取一个标记的图像文件夹。
# 你也可以创建你自己的``数据集``的子类。
# 
# 当我们实例化我们的数据集时，我们需要告诉它一些事情：
#
# -  文件系统路径是我们希望数据所存储的位置。
# -  我们是否使用这个数据集进行训练；大多数数据集将被分成训练和测试子集。
# -  如果我们还没有下载数据集，我们是否想下载它。
# -  我们要对数据进行什么样的转换。
# 
# 一旦你的数据集准备好了，你就可以把它交给 ``DataLoader``:
# 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


##########################################################################
# ``Dataset`` 的子类封装了对数据的访问，并且特定于它所服务的数据类型。
# ``DataLoader`` 对数据 *一无所知* ，而是将 ``Dataset`` 所提供的输入张量按照你指定的参数组织成批次。
# 
# 在上面的例子中，我们要求 ``DataLoader`` 从 ``trainset`` 中提供4张图片组成的批次，
# 并将它们的顺序打乱 (``shuffle=True``) ，我们告诉它启动两个workers从磁盘加载数据。
# 
# 将你的``DataLoader`` 提供的batches可视化：
# 

import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # 未正则化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 获取一些随机的训练图像
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 运行上面的单元格，应该可以看到四个图像，以及每个图像的正确标签。
# 
# 训练你的PyTorch模型
# ---------------------------
#
# 教程从视频的 `17:10 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=1030s>`__开始。
#
# 让我们把所有的片段整合在一起，训练一个模型：
#

#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#########################################################################
# 首先，我们需要训练和测试数据集。
# 如果你还没有，运行下面的单元格以确保数据集被下载。(这可能需要一些时间）。
# 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################
# 我们运行并检查 ``DataLoader`` 的输出:
# 

import matplotlib.pyplot as plt
import numpy as np

# 用于显示图像的函数


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


##########################################################################
# 这就是我们要训练的模型。如果它看起来很熟悉，那是因为它是LeNet的一个变种,适用于三通道图像。
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


######################################################################
# 最后我们还需要一个损失函数和一个优化器：
# 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


##########################################################################
# 损失函数，正如本视频前面所讨论的，用来衡量模型的预测离我们的理想输出有多远。
# 交叉熵损失是像我们这样的分类模型的一个典型损失函数。
# 
# **优化器** 用于驱动学习。在这里，我们创建了一个优化器，实现了随机梯度下降，这是一种更直接的优化算法。
# 除了算法的参数，如学习率(``lr``) 和动量（**momentum**），我们还传入 ``net.parameters()``，
# 它是模型中所有学习权重的集合--这就是优化器所调整的。
# 
# 最后，所有这些都被集合到训练循环中。运行这个单元，它可能需要几分钟的时间来执行：
# 

for epoch in range(2):  # 循环训练数据集多次

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获得输入
        inputs, labels = data

        # 将参数梯度置0
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印数据
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个mini-batches打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


########################################################################
# 在这里，我们只循环了 **2个训练周期** (第一行) --即对训练数据集传递两次。
# 每一次都有一个内循环，对**训练数据进行迭代** (line 4)，提供成批的转换后的输入图像和它们的正确标签。
# 
# 将**梯度归零** (第九行) 是一个重要步骤。梯度是在一个批次中积累的；如果我们不为每一个批次重置梯度，
# 它们将不断积累，这将提供错误的梯度值，使学习无法进行。
# 
# 在第12行，我们**要求模型对这个批次进行预测**。在接下来的第13行，
# 我们计算损失-- `输出`（模型预测）和 `标签`（正确输出）之间的差异。
# 
# 在第14行，我们进行 `backward()传递` ，并计算梯度，以指导学习。
# 
# 在第15行，优化器执行了一个学习步骤--它使用来自 `backward()` 调用的梯度，将学习权重向它认为会减少损失的方向推进。
# 
# 循环的其余部分会对世代数、已完成的训练实例的数量以及训练循环中得到的损失做一些简单的输出。
# 
# 当你运行上面的单元时，你应该看到这样的东西：
# 
# ::
# 
#    [1,  2000] loss: 2.235
#    [1,  4000] loss: 1.940
#    [1,  6000] loss: 1.713
#    [1,  8000] loss: 1.573
#    [1, 10000] loss: 1.507
#    [1, 12000] loss: 1.442
#    [2,  2000] loss: 1.378
#    [2,  4000] loss: 1.364
#    [2,  6000] loss: 1.349
#    [2,  8000] loss: 1.319
#    [2, 10000] loss: 1.284
#    [2, 12000] loss: 1.267
#    Finished Training
# 
# 我们可以注意到，损失是单调下降的，表明我们的模型在训练数据集上的性能在持续提高。
# 
# 作为最后一步，我们应该检查模型是否真的在进行一般性学习，而不是简单地 "记忆 "数据集。
# 这被称为**过拟合（overfitting）**，通常表明数据集太小（没有足够的例子进行一般性学习），
# 或者模型的学习参数超过了它对数据集正确建模的需要。
# 
# 这就是数据集被分成训练和测试集的原因--为了测试模型的通用性，我们要求它对没有训练过的数据进行预测：
# 

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('网络在10000张测试图片的准确率： %d %%' % (
    100 * correct / total))


#########################################################################
# 如果你一直跟着教程，你可以看到模型大概有50%的准确率。这是并不算最先进的，
# 但它远远好于我们从随机输出中期望的10%的准确性。这表明模型中确实发生了一些一般性的学习。
# 
