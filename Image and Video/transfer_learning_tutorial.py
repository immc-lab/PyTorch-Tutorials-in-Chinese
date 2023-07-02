# -*- coding: utf-8 -*-
"""
计算机视觉的迁移学习教程
==============================================
**作者**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

在本教程中，你将学习如何使用迁移学习训练一个卷积神经网络进行图像分类。你可以在
 `cs231n notes <https://cs231n.github.io/transfer-learning/>`__
笔记中阅读更多关于转移学习的内容。
引用这些说明，

    在实践中，很少有人从头开始训练整个卷积网络（随机初始化），因为拥有足够大的数据集是比较罕见的。
    相反，常见的做法是在一个非常大的数据集上预训练一个卷积网络（例如ImageNet，它包含120万张图片，
    有1000个类别），然后将卷积网络作为初始化或固定的特征提取器用于感兴趣的任务。

以下是两个主要的转移学习方案：

-  **对ConvNet进行微调**: 我们不采用随机初始化，而是用一个预训练的网络来初始化网络，
   就像在imagenet 1000数据集上训练的那个网络。训练的其他部分和平常一样。
-  **作为固定特征提取器的ConvNet**: 在这里，我们将冻结所有网络的权重，除了最后一个全连接层。
   最后一个全连接层被替换成一个新的具有随机权重的层，并且只对这一层进行训练。

"""
# 许可证: BSD
# 作者： Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # 互动模式

######################################################################
# 加载数据
# ---------
#
# 我们将使用torchvision和torch.utils.data包来加载数据。
#
# 我们现在要解决的问题是训练一个模型来对***蚂蚁和***蜜蜂进行分类。
# 我们有大约120张蚂蚁和蜜蜂的训练图像。每个类别有75张验证图像。
# 通常情况下，如果从头开始训练，因为这是一个非常小的数据集所以用来归纳有点困难。
# 但是我们使用的是迁移学习，所以我们应该能够很好地归纳。
#
# 这个数据集是imagenet的一个非常小的子集。
#
# .. 小贴士 ::
#    从 `这里 <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_ 下载数据并解压到当前目录。

# 用于训练的数据扩充和规范化
# 只是规范化的验证
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# 可视化图像
# ^^^^^^^^^^^^^^^^^^^^^^
# 让我们把一些训练图像可视化，以便理解数据的增强。

def imshow(inp, title=None):
    """显示张量的图像。"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一下，更新plots


# 获取一个批次训练数据
inputs, classes = next(iter(dataloaders['train']))

# 根据批次制作一个网格
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


######################################################################
# 训练模型
# ------------------
#
# 现在，让我们写一个一般的函数来训练一个模型。下面，我们将举例说明：
#
# -  安排学习率
# -  保存最佳模型
#
# 在下文中，参数 ``scheduler`` 是一个来自 ``torch.optim.lr_scheduler`` 的LR调度器对象。


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个历时有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 将模型设置为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 对数据进行迭代。
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 参数梯度为零
                optimizer.zero_grad()

                # forward
                # 只在训练时追踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段，才 backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计数据
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# 模型预测的可视化
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 显示几个图像的预测的通用函数
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# 对ConvNet进行微调
# ----------------------
#
# 加载一个预训练的模型并重置最后的全连接层。
#

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# 这里每个输出样本的大小被设定为2。
# 另外，它可以被概括为 ``nn.Linear(num_ftrs, len(class_names))``。
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# 观察到所有的参数都要优化
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 每7个历时的衰减LR为0.1倍
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# 训练和评估
# ^^^^^^^^^^^^^^^^^^
#
# 在CPU上应该需要大约15-25分钟。但在GPU上，它需要不到一分钟。
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

visualize_model(model_ft)


######################################################################
# 作为固定特征提取器的ConvNet
# ----------------------------------
#
# 在这里，我们需要冻结除最后一层以外的所有网络。我们需要设置 ``requires_grad = False``
# 来冻结参数，这样，梯度就不会在 ``backward()``中计算了。
#
# 你可以在`这里 <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__阅读更多关于这个问题的文档。
#
#

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# 新建模块的参数默认为requires_grad=True
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 观察一下，与之前相比，只有最后一层的参数被优化。
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 每7个历时的衰减LR为0.1倍
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# 训练和评估
# ^^^^^^^^^^^^^^^^^^
#
# 在CPU上，与之前的方案相比，这将花费大约一半的时间。这是预料之中的，因为大部分网络的梯度不需要计算。
# 然而，需要计算前向。
#

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()

######################################################################
# 更进一步的学习
# -----------------
#
# 如果你想了解更多关于转移学习的应用，请查看我们的《计算机视觉的量化转移学习教程》。
# <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
#
#

