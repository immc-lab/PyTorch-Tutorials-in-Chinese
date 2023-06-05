"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
**Datasets & DataLoaders** ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Datasets & DataLoaders
===================

"""

#################################################################
# 在做数据集处理工作时，处理数据样本的代码可能会变得杂乱无章且难以维护。为了提高数据集代码的可读性和规范数据集代码,我们将我们的数据集代码能够与我们的模型训练代码解耦。
# PyTorch提供了两个数据基元：``torch.utils.data.DataLoader``和``torch.utils.data.Dataset``，这两个数据基元允许你使用预先加载的数据集以及你自己的数据。
# ``Dataset``存储了样本及其相应的标签，而``DataLoader``在``Dataset``周围包裹了一个可迭代的数据集，以便能够方便地访问这些样本。
#
# PyTorch领域库提供了一些子类为``torch.utils.data.Dataset``的预加载的数据集（如FashionMNIST），并实现了针对特定数据的功能。这些可以用来为你的模型建立原型和基准。
# 你可以在这里找到它们: `Image Datasets <https://pytorch.org/vision/stable/datasets.html>`_、
# `Text Datasets  <https://pytorch.org/text/stable/datasets.html>`_ 和
# `Audio Datasets <https://pytorch.org/audio/stable/datasets.html>`_
#

############################################################
# 加载数据集
# -------------------
#
# 下面是一个如何从TorchVision加载 `Fashion-MNIST <https://research.zalando.com/project/fashion_mnist/fashion_mnist/>`_ 数据集的例子。
# Fashion-MNIST是一个由60,000个训练实例和10,000个测试实例组成的Zalando的文章图像数据集。每个例子包括一个28×28的灰度图像和10个类别中的一个相关标签。
#
# 我们用以下参数加载 `FashionMNIST Dataset <https://pytorch.org/vision/stable/datasets.html#fashion-mnist>`_ :
#  - ``root`` 是存储训练/测试数据的路径。
#  - ``train`` 指定训练或测试数据集。
#  - ``download=True`` 如果``root``下没有数据，则从网上下载。
#  - ``transform`` 和 ``target_transform`` 指定特征和标签的转换。


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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


#################################################################
# 迭代和可视化数据集
# -----------------
#
# 我们可以像列表一样手动索引 ``Datasets`` : ``training_data[index]``。
# 我们使用 ``matplotlib`` 来可视化训练数据中的一些样本。

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#################################################################
# ..
#  .. figure:: /_static/img/basics/fashion_mnist.png
#    :alt: fashion_mnist


######################################################################
# --------------
#

#################################################################
# 为你的文件创建一个自定义数据集
# ---------------------------------------------------
#
# 一个自定义的Dataset类必须实现三个函数 : `__init__`、 `__len__` 和 `__getitem__`。
# 在``img_dir``目录下，他们相应的标签存储在CSV文件``annotations_file``中。
#
# 在接下来的章节中，我们将对这些功能中的每一步进行分解。


import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#################################################################
# __init__
# ^^^^^^^^^^^^^^^^^^^^
#
# 在实例化数据集对象时，__init__函数会运行一次。我们初始化包含图像的目录、注释文件和两种转换（在下一节有更详细的介绍）。
#
# labels.csv文件例如: ::
#
#     tshirt1.jpg, 0
#     tshirt2.jpg, 0
#     ......
#     ankleboot999.jpg, 9


def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform


#################################################################
# __len__
# ^^^^^^^^^^^^^^^^^^^^
#
# 函数__len__返回我们数据集中的样本数。
#
# 例如:


def __len__(self):
    return len(self.img_labels)


#################################################################
# __getitem__
# ^^^^^^^^^^^^^^^^^^^^
#
# __getitem__ 函数从数据集中给定的下标index``idx``加载并返回一个样本。
# 基于这个下标，该函数可以识别图像的存储位置,将图像的存储位置用``read_image``函数转换为一个张量,
# 并在``self.img_labels``中的csv数据检索相应的标签，对他们调用变换（transform）
# 函数（如果适合的话），并在一个元组中返回张量图和相应的标签。

def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label


######################################################################
# --------------
#


#################################################################
# 使用DataLoaders为训练准备数据
# -------------------------------------------------
# ``Dataset`` 一次一个样本地检索我们数据集的特征和标签。 在训练模型时，我们通常希望以"minibatches"的形式传递样本，
# 在每个epoch打乱数据以减少模型的过拟合，并使用python的``multiprocessing``来加速数据检索。
#
# ``DataLoader`` 是一个可迭代的对象，它用一个简单的API为我们抽象了这种复杂性。

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

###########################
# 遍历数据加载器
# --------------------------
#
# 我们已经将该数据集加载到 ``DataLoader`` 中，并可以根据需要迭代该数据集。
# 下面的每次迭代都会返回一批 ``train_features`` 和 ``train_labels`` (分别包括``batch_size=64``特征和标签)。
# 因为我们指定了shuffle=True，所以在我们遍历所有批次后，数据会被打乱（要想对数据加载顺序进行更精细的控制，
# 请看 `Samplers <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_)。

# 显示图像和标签。
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

######################################################################
# --------------
#

#################################################################
# 进一步学习
# --------------
# - `torch.utils.data API <https://pytorch.org/docs/stable/data.html>`_
