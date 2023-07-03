# -*- coding: utf-8 -*-
"""
DCGAN 教程
==============

**作者**: `Nathan Inkawhich <https://github.com/inkawhich>`__

"""


######################################################################
# 简介
# ------------
# 本教程通过一个例子来对 DCGANs 进行介绍。
# 我们将会训练一个生成对抗网络（GAN）用于在展示了许多真正的名人的图片后产生新的名人。
# 这里的大部分代码来自 `pytorch/examples <https://github.com/pytorch/examples>`__ 中的 DCGAN 实现，
# 本文档将对实现进行进行全面 的介绍，并阐明该模型的工作原理以及为什么如此。
# 但是不需要担心，你并不需要事先了解 GAN，但可能需要花一些事件来推理一下底层 实际发生的事情。
# 此外，为了有助于节省时间，最好是使用一个GPU，或者两个。让我们从头开始。

# 
# 生成对抗网络
# -------------------------------
# 
# 什么是 GAN？
# ~~~~~~~~~~~~~~
#
# GANs是用于 DL （Deep Learning）模型去捕获训练数据分布情况的框架，以此我们可以从同一分布中生成新的数据。
# GANs是有Ian Goodfellow 于2014年提出，并且首次在论文 `Generative Adversarial
# Nets <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__中描述。
# 它们由两个不同的模型组成，一个是*生成器*，一个是*判别器*。生成器的工作是产生看起来像训练图像的‘假’图像；
# 判别器的工作是 查看图像并输出它是否是真实的训练图像或来自生成器的伪图像。
# 在训练期间，产生器不断尝试通过产生越来越好的假动作来超越判别器， 而判别器则是为了更好地检测并准确地对真实和假图像进行分类。
# 这个游戏的平衡是当生成器产生完美的假动作以使假图像看起来像是来自 训练数据，而判别器总是猜测生成器输出图像为真或假的概率为50%。
# 
# 现在，我们开始定义在这个教程中使用到的一些符号。
# 判别器的符号定义：
# 设 :math:`x`表示代表一张图像的数据， :math:`D(x)` 是判别器网络，它输出来自训练数据而不是生成器的（标量）概率。
# 这里，由于我们处理图像， :math:`D(x)`的输入是 CHW 大小为3x64x64的图像。
# 直观地，当:math:`x`来自训练数据时:math:`D(x)`应该是 HIGH ，而当:math:`x`来自生成器时:math:`D(x)`应该是 LOW。也可以被认为是传统的二元分类器。
# 
# 生成器的符号定义：
# 对于生成器的符号，让:math:`z`是从标准正态分布中采样的潜在空间矢量，:math:`G(z)`表示将潜在向量:math:`z`映射到数据空间的生成器函数，
# :math:`G`的目标是估计训练数据来自什么分布(:math:`p_{data}`) ，
# 以便它可以 根据估计的分布(:math:`p_g`)生成假样本。
#
# 因此，:math:`D(G(z))`是生成器:math:`G`的输出是真实图像的概率（标量）。正如Goodfellow 的论文中所描述的，:math:`D` 和 :math:`G`玩一个极小极大的游戏，
# 其中:math:`D` 试图最大化它正确地分类真实数据和假样本(:math:`logD(x)`)的概率，并且:math:`G`试图最小化:math:`D`预测其输出是假的概率(:math:`log(1-D(G(z)))`)。
# 从论文来看，GAN 损失函数是:
# .. math:: \underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(z)))\big]
#
# 理论上，这个极小极大游戏的解决方案是:math:`p_g = p_{data}`，如果输入的是真实的或假的，则判别器会随机猜测。
# 然而，GAN 的收敛理论仍在积极研究中，实际上模型并不总是训练到这一点。
# 
# 什么是 DCGAN？
# ~~~~~~~~~~~~~~~~
#
# DCGAN 是上述 GAN 的直接扩展，区别的是它分别在判别器和生成器中明确地使用了卷积和卷积转置层。
# 它首先是由Radford等人在论文 `Unsupervised Representation Learning With
# Deep Convolutional Generative Adversarial Networks <https://arxiv.org/pdf/1511.06434.pdf>`__中提出。
# 判别器由 `convolution <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`__层、
# `batch norm <https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d>`__层和
# `LeakyReLU <https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU>`__激活函数组成，它输入 3x64x64 的图像，
# 然后输出的是一个代表输入是来自实际数据分布的标量概率。
# 生成器则是由`convolutional-transpose <https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d>`__层、 batch norm 层
# 和 `LeakyReLU <https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU>`__激活函数组成。
# 它的输入是从标准正态分布中绘制的潜在向量:math:`z`，输出是 3x64x64 的 RGB 图像。
# strided conv-transpose layers 允许潜在标量转换成具有与图像相同形状的体积。
# 在本文中，作者还提供了一些有关如何设置优化器，如何计算损失函数以及如何初始化 模型权重的提示，所有这些都将在后面的章节中进行说明。
# 

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 为再现性设置随机seed
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 如果你想要新的结果就是要这段代码
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


######################################################################
# 输入
# ------
# 
# 让我们定义输入数据去运行我们的教程：
# 
# -  ``dataroot`` - 存放数据集根目录的路径。我们将在下一节中详细讨论数据集
# -  ``workers`` - 使用DataLoader加载数据的工作线程数
# -  ``batch_size`` - 训练中使用的batch大小。在DCGAN论文中batch的大小为128
# -  ``image_size`` - 用于训练的图像的空间大小。此实现默认 64x64。如果需要其他尺寸，则必须改变和 的结构。
#     有关详细信息，请参见`此处 <https://github.com/pytorch/examples/issues/70>`__。

# -  ``nc`` - 输入图像中的颜色通道数。对于彩色图像，这是参数设置为3
# -  ``nz`` - 潜在向量的长度
# -  ``ngf`` - 与通过生成器携带的特征图的深度有关
# -  ``ndf`` - 设置通过判别器传播的特征映射的深度
# -  ``num_epochs`` - 要运行的训练的epoch数量。长时间的训练可能会带来更好的结果，但也需要更长的时间
# -  ``lr`` - 学习速率。如DCGAN论文中所述，此数字应为0.0002
# -  ``beta1`` - 适用于Adam优化器的beta1超参数。如论文所述，此数字应为0.5
# -  ``ngpu`` - 可用的GPU数量。如果为0，则代码将以CPU模式运行。如果此数字大于0，它将在该数量的GPU上运行
#

# 数据集的根目录
dataroot = "data/celeba"

# 加载数据的工作线程数
workers = 2

# 训练期间的batch大小
batch_size = 128

# 训练图像的空间大小。所有图像将使用Transformer调整为此大小。
image_size = 64

# 训练图像中的通道数。对于彩色图像，这里是3
nc = 3

# 潜在向量 z 的大小(例如： 生成器输入的大小)
nz = 100

# 生成器中特征图的大小
ngf = 64

# 判别器中的特征映射的大小
ndf = 64

# 训练epochs的大小
num_epochs = 5

# 优化器的学习速率
lr = 0.0002

# 适用于Adam优化器的Beta1超级参数
beta1 = 0.5

# 可用的GPU数量。使用0表示CPU模式。
ngpu = 1


######################################################################
# 数据
# ----
#
# 在本教程中，我们将使用`Celeb-A Faces
# dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__数据集，该数据集可以在链接或`Google
#  Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__中下载。
# 数据集将下载为名为 ``img_align_celeba.zip`` 的文件。下载后，创建名为 ``celeba`` 的目录并将zip文件解压缩到该目录中。然后，将此笔记中
# 的 `dataroot`` 输入设置为刚刚创建的 ``celeba`` 目录。生成的目录结构应该是：
# 
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
#
# 这是一个重要的步骤，因为我们将使用 ``ImageFolder`` 数据集类，它要求在数据集的根文件夹中有子目录。
# 现在，我们可以创建数据集，创 建数据加载器，设置要运行的设备，以及最后可视化一些训练数据。
# 

# 我们可以按照设置的方式使用图像文件夹数据集。
# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 选择我们运行在上面的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 绘制部分我们的输入图像
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))



######################################################################
# 实现
# --------------
#
# 通过设置输入参数和准备好的数据集，我们现在可以进入真正的实现步骤。
# 我们将从权重初始化策略开始，然后详细讨论生成器，鉴别器， 损失函数和训练循环。
# 
#  权重初始化
# ~~~~~~~~~~~~~~~~~~~~~
#
# 在DCGAN论文中，作者指出所有模型权重应从正态分布中随机初始化，``mean=0``，``stdev=0.02``。
# ``weights_init`` 函数将初始化模型作为输入，并重新初始化所有卷积，卷积转置和batch标准化层以满足此标准。初始化后立即将此函数应用于模型。
# 

# 传统的权重初始化调用 ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################################
# 生成器
# ~~~~~~~~~
#
# 生成器:math:`G`用于将潜在空间矢量（:math:`z`）映射到数据空间。
# 由于我们的数据是图像，因此将:math:`z`转换为数据空间意味着最终创建与训练图像具有相同大小的RGB图像（即3x64x64）。
# 实际上，这是通过一系列跨步的二维卷积转置层实现的， 每个转换层与二维批量标准层和relu activation进行配对。
# 生成器的输出通过tanh函数输入，使其返回到:math:`[-1,1]`范围的输入数据。
# 值得 注意的是在转换层之后存在批量范数函数，因为这是DCGAN论文的关键贡献。
# 这些层有助于训练期间的梯度流动。DCGAN论文中的生成器中 的图像如下所示：
#
# .. figure:: /_static/img/dcgan_generator.png
#    :alt: dcgan_generator
#
# 请注意，我们对输入怎么设置(``nz``, ``ngf``, 和 ``nc``)会影响代码中的生成器体系结构。
# ``nz`` 是输入向量的长度， ``ngf`` 与通过生成器传播的特征图的大小有关，``nc`` 是输出图像中的通道数（对于RGB图像，设置为3）。
# 下面是生成器的代码。
#
# 生成器代码
# 生成器代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# 现在，我们可以实例化生成器并应用 ``weights_init`` 函数。查看打印的模型以查看生成器对象的结构。
# 

# 创建生成器
netG = Generator(ngpu).to(device)

# 如果需要，管理multi-gpu
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 应用 ``weights_init`` 函数随机初始化所有权重，``mean=0``, ``stdev=0.02``
netG.apply(weights_init)

# 打印模型
print(netG)


######################################################################
# 判别器
# ~~~~~~~~~~~~~
#
# 如上所述，判别器:math:`D`是二进制分类网络，它将图像作为输入并输出输入图像是真实的标量概率（与假的相反）。
# 这里，:math:`D`采用 3x64x64 的输入图像，通过一系列Conv2d，BatchNorm2d和LeakyReLU层处理它，并通过Sigmoid激活函数输出 最终概率。
# 如果问题需要，可以使用更多层扩展此体系结构，但使用strided convolution（跨步卷积），BatchNorm和LeakyReLU具有重要意义。
# DCGAN论文提到使用跨步卷积而不是池化到降低采样是一种很好的做法，因为它可以让网络学习自己的池化功能。
# 批量标准和 leaky relu函数也促进良好的梯度流，这对于:math:`G` 和 :math:`D`的学习过程都是至关重要的。
# 

#########################################################################
# 判别器代码

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# 现在，与生成器一样，我们可以创建判别器，应用 ``weights_init`` 函数，并打印模型的结构。
# 

# 创建判别器
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 应用weights_init函数随机初始化所有权重，mean= 0，stdev = 0.2
netD.apply(weights_init)

# 打印模型
print(netD)


######################################################################
# 损失函数和优化器
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 通过:math:`D` 和 :math:`G`设置，我们可以指定他们如何通过损失函数和优化器学习。我们将使用PyTorch中定义的二进制交叉熵损失（BCELoss <https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss>`__）函数：
# .. math:: \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
#
# 注意该函数如何计算目标函数中的两个对数分量（即:math:`log(D(x))` 和 :math:`log(1-D(G(z)))`。我们可以指定用于输入:math:`y`的BCE方程的哪个部分。这是在即将出现的训练循环中完成的，但重要的是要了解我们如何通过改变:math:`y`（即GT标签）来选择我们希望计算的组件。
#
# 接下来，我们将真实标签定义为1，将假标签定义为0。这些标签将在计算:math:`D` 和 :math:`G`的损失时使用，这也是原始 GAN 论文中使用的惯例。
# 最后，我们设置了两个单独的优化器，一个用于:math:`D`，一个用于:math:`G`。 如 DCGAN 论文中所述，两者都是Adam优化器，学习率为0.0002，Beta1 = 0.5。
# 为了跟踪生成器的学习进度，我们将生成一组固定的潜在 向量，这些向量是从高斯分布（即fixed_noise）中提取的。
# 在训练循环中，我们将周期性地将此fixed_noise输入到:math:`G`中，并且在迭代中我们将看到图像形成于噪声之外。


# 初始化BCELoss函数
criterion = nn.BCELoss()

# 创建一批潜在的向量，我们将用它来可视化生成器的进程
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 在训练期间建立真假标签的惯例
real_label = 1
fake_label = 0

# 为 G 和 D 设置 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


######################################################################
# 训练
# ~~~~~~~~
# 
# 最后，既然已经定义了 GAN 框架的所有部分，我们就可以对其进行训练了。
# 请注意，训练GAN在某种程度上是一种艺术形式，因为不正确 的超参数设置会导致对错误的解释很少的模式崩溃，
# 在这里，我们将密切关注`Goodfellow’s paper <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__的论文中的算法1，同时遵守`ganhacks <https://github.com/soumith/ganhacks>`__中展示的一些最佳实践。
# 也就是说，我们将“为真实和虚假”图像构建不同的 mini-batches ，并且还调整G的目标函数以最大化:math:`log(D(G(z)))`。
# 训练分为两个主要部分，第1部分更新判别器，第2部分更新生成器。
# 
# **第一部分：训练判别器**
# 
# 回想一下，训练判别器的目的是最大化将给定输入正确分类为真实或假的概率。
# 就Goodfellow而言，我们希望“通过提升其随机梯度来更新判别器”。实际上，我们希望最大化:math:`log(D(x)) + log(1-D(G(z)))`。
# 由于`ganhacks <https://github.com/soumith/ganhacks>`__的独立 mini-batch 建议，我们将分两步计算。
# 首先，我们将从训练集构建一批实际样本，向前通过:math:`D`，计算损失(:math:`log(D(x))`)，然后计算向后传递的梯度。
# 其次，我们将用当前生成器构造一批假样本，通过向前传递该 batch，计算损失，并通过反向传递累积梯度。
# 现在，随着从全实时和全实时批量累积的梯度，我们称之为Discriminator优化器的一步。
# 
# **第二部分：训练生成器**
# 
# 正如原始论文所述，我们希望通过最小化:math:`log(1-D(G(z)))`来训练生成器，以便产生更好的伪样本。
# 如上所述，Goodfellow 表 明这不会提供足够的梯度，尤其是在学习过程的早期阶段。
# 作为修复，我们希望最大化:math:`log(D(G(z)))`。在代码中，我们通过以下方式实现此目的：使用判别器对第1部分的生成器中的输出进行分类，使用真实标签： GT标签计算的损失，在向后传递中计算的梯度，
# 最后使用优化器步骤更新G的参数。使用真实标签作为损失函数的GT 标签似乎是违反直觉的，但是这允许我们使用 ``BCELoss`` 的:math:`log(x)`部分(而不是 :math:`log(1-x)` 部分) ，这正是我们想要。
#
# 最后，我们将进行一些统计报告，在每个epoch结束时，我们将通过生成器推送我们的fixed_noise batch，以直观地跟踪 训练的进度。训练的统计数据是：
# 
# -  **Loss_D** - 判别器损失计算为所有实际批次和所有假批次的损失总和 (:math:`log(D(x)) + log(1 - D(G(z)))`).
# -  **Loss_G** - 计算生成器损失 :math:`log(D(G(z)))`
# -  **D(x)** - 所有实际批次的判别器的平均输出（整批）。当变好时这应该从接近1开始，然后理论上收敛到0.5。 想想为什么会这样。
# -  **D(G(z))** - 所有假批次的平均判别器输出。第一个数字是在更新之前，第二个数字是在 更新之后。当G变好时，这些数字应该从0开始并收敛到0.5。想想为什么会这样。
# 
# **Note:** 此步骤可能需要一段时间，具体取决于您运行的epoch数以及是否从数据集中删除了一些数据。
# 

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # 对于数据加载器中的每个batch
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


######################################################################
# 结果
# -------
# 
# 最后，让我们看看我们是如何做到的。在这里，我们将看看三个不同的结果。首先，我们将看到D和G的损失在训练期间是如何变化的。
# 其次，我们将可视化在每个epoch的 fixed_noise batch中的输出。第三，我们将查看来自G的紧邻一批实际数据的一批假数据。
# 
# **损失与训练迭代**
# 
# 下面是D＆G的损失与训练迭代的关系图。
# 

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


######################################################################
# **G的过程可视化**
# 
# 记住在每个训练epoch之后我们如何在fixed_noise batch中保存生成器的输出。
# 现在，我们可以通过动画可视化G的训练进度。按播放按钮 开始动画。
# 

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


######################################################################
# **真实图像 vs 伪图像**
# 
# 最后，让我们一起看看一些真实的图像和伪图像。
# 

# 从数据加载器中获取一批真实图像
real_batch = next(iter(dataloader))

# 绘制真实图像
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 在最后一个epoch中绘制伪图像
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


######################################################################
# 进一步的工作
# ----------------
# 
# 我们已经完成了我们的整个教程，但是你可以从下面几个方向进一步探讨。 你可以：
# 
# -  训练更长时间，看看结果有多好
# -  修改此模型以获取不同的数据集，并或者更改图像和模型体系结构的大小
# -  在`这里 <https://github.com/nashory/gans-awesome-applications>`__查看其他一些很酷的GAN项目
# -  创建生成`音乐 <https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio/>`__的GAN
#

