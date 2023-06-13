"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
**Building Models** ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

Building Models with PyTorch
============================

本节课可以在youtube上观看。 `youtube <https://www.youtube.com/watch?v=OSqIP-mOWOI>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/OSqIP-mOWOI" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

``torch.nn.Module`` 和 ``torch.nn.Parameter``
----------------------------------------------

在视频中我们将要讨论一些用于构建深度学习网络的PyTorch工具。

除了 ``Parameter``，我们在视频中所讨论的类都是 ``torch.nn.Module`` 的子类。 这是PyTorch的基类，封装了特定的PyTorch模型及其组件。

``torch.nn.Module`` 的一个重要的操作是注册参数。
如果一个特定的 ``Module`` 子类具有可学习的权重，这些权重会被表达为 ``torch.nn.Parameter`` 的实例。
``Parameter`` 类是 `torch.Tensor``类具有特殊的行为的子类，当它们被分配为一个模块的属性时，它们会被添加到该模块的参数列表中。
这些参数可以通过 `Module`` 类的 ``parameters()`` 方法访问。


作为一个简单的例子，这里有一个非常简单的模型，有两个线性层和一个激活函数。
我们将创建它的一个实例，并要求它报告其参数：


"""

import torch

class TinyModel(torch.nn.Module):
    
    def __init__(self):
        super(TinyModel, self).__init__()
        
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)


#########################################################################
# 这显示了PyTorch模型的基本结构：有一个 ``__init__()`` 方法来定义模型的层和其他组件，还有一个 ``forward()`` 方法来完成计算。
# 请注意，我们可以打印该模型或其任何子模块，以了解其结构。
# 
# Common Layer Types
# ------------------
# 
# Linear Layers
# ~~~~~~~~~~~~~
#
# 最基本的神经网络层类型是 *线性* 或 *全连接层* 。
# 这个层的每个输入都会影响该层的每个输出，其影响程度由该层的权重决定。
# 如果一个模型有 *m* 个输入和 *n* 个输出，权重将是一个 *m* x *n* 矩阵。比如说：
# 

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)


#########################################################################
# 如果你用线性层的权重对 ``x`` 进行矩阵乘法，并加上偏置，你会发现你得到了输出向量 ``y``。
#
# 还有一个重要的特征需要注意：
# 当我们用 ``lin.weight`` 检查我们层的权重时，它报告自己是一个 ``Parameter``（这是Tensor的一个子类），并让我们知道它在用autograd追踪梯度。
# 这是 ``Parameter`` 区别于 ``Tensor`` 的一个默认行为。
#
# 线性层在深度学习模型中被广泛使用。
# 你最常看到它们的地方之一是在分类器模型中，通常会在最后有一个或多个线性层，其中最后一层会有 *n* 个输出，其中 *n* 是分类器处理的类别数量。
# 
# Convolutional Layers
# ~~~~~~~~~~~~~~~~~~~~
#
# *卷积层* 是为了处理具有高度空间相关性的数据而建立的。
# 它们在计算机视觉中非常常用，在那里它们可以检测到紧密的特征分组，并将其组成更高级别的特征。
# 它们也出现在其他场合--例如，在NLP应用中，一个词的直接上下文（即序列中邻近的其他词）可以影响一个句子的含义。
# 
# 我们在之前的视频中看到了卷积层在LeNet5中的作用：
# 

import torch.functional as F

import torch.functional as F
import torch.nn as nn


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
        x = x.view(-1, self.num_flat_features(x))  # 改变x的形状变为（batch, 其他所有维度相乘）
        x = F.relu(self.fc1(x))  # relu 是一个激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # size拿到了除批量维度（batch)外x的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


##########################################################################
# 让我们分解模型中的卷积层看看其中发生了什么，我们从 ``conv1`` 开始：
# 
# - LeNet5是为了接收1x32x32的黑白图像。**卷积层构造函数的第一个参数是输入通道的数量。** 这里是1。如果我们建立这个模型是为了观察3个颜色的通道，那么它就是3。
# - 卷积层就像一个窗口，在图像上扫描，寻找它所识别的模式。这些模式被称为 *特征* ，卷积层的一个参数是我们希望它学习的特征数量。**这是构造函数的第二个参数，是输出特征的数量。** 这里，我们要求这个层学习6个特征。
# - 在上面，我把卷积层比喻成一个窗口--但这个窗口有多大？ **第三个参数是窗口或核的大小** 。
#   这里，"5 "意味着我们选择了一个5x5的核。(如果你想要一个高度与宽度不同的核，你可以为这个参数指定一个元组--例如，``(3, 5)``来得到一个3x5的卷积核）。
#
# 卷积层的输出是一个 *激活图* --输入张量中特征存在的空间表示。
# ``conv1`` 会给我们一个6x28x28的输出张量；6是特征的数量，28是我们图的高度和宽度。
# (28来自于这样一个事实：当在32像素的行上扫描一个5像素的窗口时，只有28个有效位置）。
#
# 然后，我们将卷积的输出通过一个ReLU激活函数（后面会有更多关于激活函数的内容），然后通过最大池化（max pooling）层。
# 最大池化层将激活图中彼此相近的特征集中起来。
# 它通过减少张量来做到这一点，将输出中的每一个2x2的单元组合并成一个单元，并为该单元分配进入该单元的4个单元的最大值。
# 这样我们得到了一个低分辨率的激活图，尺寸为6x14x14。
#
# 我们的下一个卷积层，``conv2```，期望有6个输入通道（对应于第一层所寻求的6个特征），有16个输出通道和一个3x3卷积核。
# 它输出了一个16x12x12的激活图，这个激活图又被一个最大池化层减少到16x6x6。
# 在将这个输出传递给线性层之前，它被重塑为一个16*6*6=576元素的向量，供下一层使用。
#
# 有用于处理一维、二维和三维张量的卷积层。卷积层构造函数还有很多可选参数，包括输入中的步长（stride）长度（例如，每二个或每三个位置扫描一次）、填充（以便你可以扫描到输入的边缘）等等。
# 更多信息请参见`文档 <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__。
# 
# Recurrent Layers
# ~~~~~~~~~~~~~~~~
#
# *循环神经网络*（或称 *RNN）* 用于处理顺序数据--从科学仪器的时间序列测量到自然语言句子到DNA核苷酸的任何东西。
# 一个RNN通过维护一个 *隐藏的状态* 来实现这一目的，该状态作为一种记忆，用于记忆迄今为止它在序列中所看到的内容。
#
# RNN层的内部结构--或其变种LSTM（长短时记忆）和GRU（门控递归单元）--相当复杂，超出了本视频的范围，但我们将向你展示一个基于LSTM的部分语音标记器（一种分类器，告诉你一个词是否是名词、动词等）是什么样的：
# 

class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


########################################################################
# 构建器包含四个参数：
# 
# -  ``vocab_size`` 是输入词汇中单词的数量。每个词都是 ``vocab_size`` 维度空间中的一个独热（one-hot）向量（或单位向量）。
# -  ``tagset_size`` 是输出集标签（tags）的数量。
# -  ``embedding_dim`` embedding_dim是词汇的 *嵌入* 空间的大小。嵌入将词汇映射到一个低维空间上，在这个空间里，具有相似含义的词汇是紧密相连的。
# -  ``hidden_dim`` 是LSTM的记忆空间的大小。.
#
# 输入将是一个句子，其中的单词表示为独热向量的索引。然后，嵌入层将把这些内容映射到一个 ``embedding_dim`` 维的空间。
# LSTM接受这个嵌入序列并对其进行迭代，得到一个长度为 ``hidden_dim`` 的输出向量。
# 最后的线性层充当分类器；对最后一层的输出应用 `log_softmax()``，将输出转换为一个标准化的估计概率集，即一个给定的词映射到一个给定的标签。
# 
# 如果你想看看这个网络的运行情况，请查看pytorch.org上的 `Sequence
# Models 和 LSTM
# Networks <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>`__。
# 
# Transformers
# ~~~~~~~~~~~~
# 
# *Transformers* 是一种多用途的网络，在NLP领域，带来了像BERT这样十分先进的模型。
# 对Transformers架构的讨论超出了本视频的范围，但PyTorch有一个 ``Transformer`` 类，
# 允许你定义Transformer模型的整体参数--注意头的数量、编码器和解码器层的数量、丢弃和激活函数等（只要参数正确，你甚至可以用这个单一的类来建立BERT模型）。
# ``torch.nn.Transformer`` 类也有类来封装各个组件（``TransformerEncoder``、``TransformerDecoder``）和子组件（``TransformerEncoderLayer``、``TransformerDecoderLayer``）。
#  详情请查看transformer类的 `文档 <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__ ，
#  以及pytorch.org上的相关`教程 <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`__。
# 
# Other Layers and Functions
# --------------------------
# 
# Data Manipulation Layers
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 还有一些层类型在模型中执行重要功能，但本身不参与学习过程。
#
# **最大池化（Max pooling）**（以及它的孪生最小池化（min pooling））通过组合单元格来减少张量，并将输入单元格的最大值分配给输出单元格。比如说：
# 

my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


#########################################################################
# 如果你仔细观察上面的数值，你会发现最大池化的输出中的每个数值都是6x6输入的每个象限的最大值。
#
# **归一化层（Normalization layers）** 在将一个层的输出送入另一个层之前，对其进行重集中和归一化。
# 对中间张量进行集中和缩放有很多好处，比如让你使用更高的学习率，而不会出现梯度爆炸/消失。
# 

my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())



##########################################################################
##%% md 运行上面的单元格，我们给输入张量添加了一个大的缩放因子和偏移量；你应该看到输入张量的 `mean()`` 在15附近。
# 在通过归一化层运行后，你可以看到数值变小了，并且围绕着零进行分组--事实上，平均值应该非常小（>1e-8）。
#
# 这是有好处的，因为许多激活函数（下文将讨论）在0附近有最大的梯度，但有时会因输入远离0而导致梯度消失或爆炸。
# 保持数据以最陡峭的梯度区域为中心，往往意味着更快、更好的学习和更高的可行的学习率。
#
# **丢弃层（Dropout layers）** 是一个鼓励模型中的稀疏表征的工具--也就是说，促使它用更少的数据进行推理。
#
# 丢弃层的工作方式是在 *训练期间* 随机设置输入张量的一部分--丢弃层在推理时是关闭的。
# 这迫使模型针对这个被屏蔽或减少的数据集进行学习。比如说：
# 

my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))


##########################################################################
# 上面，你可以看到dropout对样本张量的影响。你可以使用可选的 ``p`` 参数来设置单个权重被丢弃的概率；
# 如果你不指定，则默认为0.5。
# 
# Activation Functions
# ~~~~~~~~~~~~~~~~~~~~
#
# 激活函数使深度学习成为可能。一个神经网络实际上是一个有许多参数的程序，用来 *模拟一个数学函数* 。
# 如果我们所做的只是通过层权重重复多个张量，我们只能模拟 *线性函数* ；
# 此外，拥有许多层也没有意义，因为整个网络将减少可以简化为一个单一的矩阵乘法。
# 在层之间插入 *非线性* 激活函数，才能让深度学习模型模拟任何函数，而不仅仅是线性函数。
# 
# ``torch.nn.Module`` 有封装所有主要激活函数的对象，包括ReLU及其许多变体、Tanh、Hardtanh、sigmoid等。
# 它还包括其他函数，如Softmax，这些函数在模型的输出阶段最为有用。
# 
# Loss Functions
# ~~~~~~~~~~~~~~
#
# 损失函数告诉我们一个模型的预测离正确答案有多远。
# PyTorch包含各种损失函数，包括常见的MSE（均方误差=L2范数）、交叉熵损失和负似然损失（对分类器很有用），以及其他函数。
# 
