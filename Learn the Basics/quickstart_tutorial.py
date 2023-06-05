"""
`学习基础知识 <intro.html>`_ ||
`快速开始 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集和数据加载器 <data_tutorial.html>`_ ||
`变换  <transforms_tutorial.html>`_ ||
`创建模型 <buildmodel_tutorial.html>`_ ||
**自动求导** ||
`优化 <optimization_tutorial.html>`_ ||
`保存和加载模型 <saveloadrun_tutorial.html>`_

快速开始
===================
本节介绍了机器学习中常见任务的API。请参考每一节中的链接来深入了解。

与数据一起工作
-----------------
PyTorch有两个`处理数据的原语 <https://pytorch.org/docs/stable/data.html>`_:
``torch.utils.data.DataLoader`` 和 ``torch.utils.data.Dataset`` 。
``Dataset`` 存储样本和它们相应的标签。 ``DataLoader`` 在 ``Dataset`` 周围包装了一个可迭代的东西.

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

######################################################################
# PyTorch提供了特定领域的库，如`TorchText <https://pytorch.org/text/stable/index.html>`_、
# `TorchVision <https://pytorch.org/vision/stable/index.html>`_、
# 和 `TorchAudio <https://pytorch.org/audio/stable/index.html>`_，
# 所有这些都包括数据集。 在本教程中，我们将使用一个TorchVision数据集。
#
# ``torchvision.datasets`` 模块包含许多现实世界视觉数据的`Dataset`对象，如CIFAR，COCO
# (`full list here <https://pytorch.org/vision/stable/datasets.html>`_)。在本教程中，我们使用FashionMNIST数据集。
# 每个TorchVision ``Dataset`` 包括两个参数： ``transform`` 和 ``target_transform`` ，这两个参数分别用来修改样本和标签。

# 从开放数据集下载训练数据。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 从开放数据集下载测试数据。
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

######################################################################
# 我们将 ``Dataset`` 作为参数传递给 ``DataLoader`` 。这在我们的数据集上包裹了一个可迭代的数据集，
# 并支持自动批处理、采样、洗牌和多进程数据加载。这里我们定义了一个64的批处理量，也就是说，每个元素都将返回64个特征和标签的批次。

batch_size = 64

# 创建数据加载器。
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

######################################################################
# 阅读更多关于`在PyTorch中加载数据 <data_tutorial.html>`_。
#

######################################################################
# --------------
#

################################
# 创建模型
# ------------------
# 为了在PyTorch中定义一个神经网络，我们创建一个继承自`nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
# 的类。我们在 ``__init__`` 函数中定义网络的层，并在 ``forward`` 函数中指定数据如何通过网络。
# 为了加速神经网络的操作，我们将其移至GPU或MPS（如果有的话）。

# 获取cpu、gpu或mps设备进行培训。
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

######################################################################
# 阅读更多关于`在PyTorch中构建神经网络 <buildmodel_tutorial.html>`_.
#


######################################################################
# --------------
#


#####################################################################
# 优化模型参数
# ----------------------------------------
# 为了训练一个模型，我们需要一个`损失函数 <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# 和一个 `优化器 <https://pytorch.org/docs/stable/optim.html>`_.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


#######################################################################
# 在一个单一的训练循环中，模型对训练数据集（分批送入）进行预测，然后反向传播预测误差来调整模型的参数。

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# 为了确保模型在学习，我们还根据测试数据集检查模型的性能。

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################################################################
# 训练过程是通过几个迭代(*历时*)进行的。在每个历时中，模型学习参数以做出更好的预测。
# 我们在每个历时中打印模型的准确性和损失；我们希望准确率增加，损失减少。

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

######################################################################
# 阅读更多关于`训练你的模型 <optimization_tutorial.html>`_.
#

######################################################################
# --------------
#

######################################################################
# 保存模型
# -------------
# 保存模型的一个常见方法是序列化内部状态字典（包含模型参数）。

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



######################################################################
# 加载模型
# ----------------------------
#
# 加载一个模型的过程包括重新创建模型结构和在模型结构中加载状态字典。

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

#############################################################
# 这个模型现在可以用来进行预测。

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


######################################################################
# 阅读更多关于 `保存和加载你的模型 <saveloadrun_tutorial.html>`_.
#
