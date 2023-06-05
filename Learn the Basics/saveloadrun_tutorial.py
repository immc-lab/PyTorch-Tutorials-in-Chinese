"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
**Save & Load Model**

保存和加载模型
============================

在这一节中我们将关注如何通过保存，加载和运行模型预测来保持模型的状态。
"""

import torch
import torchvision.models as models


#######################################################################
# 保存和加载模型权重
# --------------------------------
# Pytorch模型在内置的状态字典 ``state_dict`` 保存学习到的参数。
# 这些参数可以通过 ``torch.save``方法保存:

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# 要加载模型权重，首先需要创建一个相同模型的实例（instance），
# 并且使用 ``load_state_dict()`` 方法加载参数。

model = models.vgg16() # 我们不指定权重 ``weights``，即创建一个没训练过的模型
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

###########################
# .. note:: 请确保在推理前调用 ``model.eval()`` 方法，将丢弃（dropout）和批归一化（batch normalization）层设置为评估模式。如果不这样做，将产生不一致的推理结果。
#######################################################################
# 保存和加载带形状的模型
# -------------------------------------
# 在加载模型权重时，我们需要先将模型类实例化，
# 因为该类定义了网络的结构。我们可能想把这个类的结构和模型一起保存，
# 在这种情况下，我们可以把 ``model``（而不是 ``model.state_dict()``）传给保存函数：

torch.save(model, 'model.pth')

########################
# 我们可以这样来加载模型:

model = torch.load('model.pth')

########################
# .. note:: 这种方法在序列化模型时使用Python [pickle](https://docs.python.org/3/library/pickle.html)模块，因此在加载模型时它依赖于可用的实际类定义。

#######################
# 相关教程
# -----------------
# `在PyTorch中保存和加载一个通用保存点（checkpoint） <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`_
