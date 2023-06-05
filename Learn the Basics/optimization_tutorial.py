"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
**Optimization** ||
`Save & Load Model <saveloadrun_tutorial.html>`_

优化模型参数
===========================
现在我们有了模型和数据，是时候通过优化参数对我们的模型进行训练、验证和测试了。
训练模型是一个迭代的过程；在每一次迭代中模型会预测一个输出，并对它的输出计算一个误差（*损失*），
收集这些相关参数的误差的导数（如上一节所示），并使用梯度下降**优化**这些参数。
关于这个过程更的详细介绍，请看3Blue1Brown的 "反向传播 "视频 <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__。


之前的代码
-----------------
我们从之前的数据集&数据加载器和建立模型章节中加载代码。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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

model = NeuralNetwork()


##############################################
# 超参数
# -----------------
#
# 超参数是让你控制模型优化过程的可调参数。不同的超参数值可能会影响模型的训练和收敛率
# （[阅读更多](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)有关超参数调整的内容）。

# 我们定义了如下的超参数用于训练：
# - **Number of Epoch（世代数）** 迭代数据集的次数
# - **Batch Size（批量大小）** 参数更新前通过网络传播的数据样本数
# - **Learning Rate（学习率）**在每个批次/阶段更新模型参数的程度。较小的值产生缓慢的学习速度，而较大的值可能导致训练期间的不可预测的行为。

learning_rate = 1e-3
batch_size = 64
epochs = 5



#####################################
# 优化循环
# -----------------
# 
# 当我们设置好我们的超参数，我们就可以通过优化循环训练和优化我们的模型。每个优化循环的迭代乘以一个**世代(epoch)**。

# 每个世代由两个主要的部分组成：
# - **训练循环** - 迭代训练集并且尝试通过优化参数收敛模型。
# - **验证/测试循环** - 迭代测试集并查看模型性能是否有所提升。

# 让我们简单地熟悉一下训练循环中使用的一些概念。
# 在这里看看优化循环的[完整实现](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-impl-label)。

#
# 损失函数
# ~~~~~~~~~~~~~~~~~
#
# 当遇到一些训练数据时，我们未经训练的网络给出的答案很可能是错误的。
# **损失函数(Loss funcation)** 衡量的是模型输出的结果与目标值的不相似程度，它是我们在训练期间想要最小化的损失函数。
# 为了计算损失，我们使用给定数据样本的输入进行预测，并与真实数据标签值进行比较。

# 常用的损失函数包括回归任务的[nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)(均方误差)，
# 分类任务的[nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)(负对数似然)。
# [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)结合了`nn.LogSoftmax`和`nn.NLLLoss`。

# 我们将模型的输出对数传递给`nn.CrossEntropyLoss`，它将对对数进行归一化处理并计算预测误差。


# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()



#####################################
# 优化器
# ~~~~~~~~~~~~~~~~~
#
# 优化是在每一步训练中调节模型参数以减少模型误差的过程。**优化算法** 定义了这个过程如何被执行（在这个例子中我们使用了随机梯度下降）。
# 所有的优化逻辑都被封装在优化器对象中。在这里，我们使用SGD优化器；
# 此外，PyTorch中还有许多[不同的优化器](https://pytorch.org/docs/stable/optim.html)，如ADAM和RMSProp，它们对不同类型的模型和数据有更好的效果。

# 我们通过注册需要训练的模型参数来初始化优化器，并传入学习率超参数。

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#####################################
# 在训练循环中，优化会发生在三个步骤中：
# - 调用 ``optimizer.zero_grad()`` 来重新设置模型参数的梯度。梯度默认为累加；为了防止重复计算，我们在每次迭代时都明确地将其归零。
# - 通过调用 ``loss.backward()`` 对预测损失进行反向传播。PyTorch将损失的梯度与每个参数结合起来。
# - 一旦我们有了梯度，我们就调用 ``optimizer.step()``，通过后向传递中收集的梯度来调整参数。

########################################
# .. _full-impl-label:
#
# 完整实现
# -----------------------
# 我们定义了 `train_loop` 和 `test_loop` ，`train_loop`负责循环我们的优化代码，
# `test_loop` 负责根据测试数据评估模型的性能。

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


########################################
# 我们初始化损失函数和优化器，并将其传递给 ``train_loop`` 和 ``test_loop``。
# 增加epochs的数量来跟踪模型的改进性能。

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



#################################################################
# 阅读更多
# -----------------------
# - `损失函数 <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `暖启动训练模型 <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_
#
