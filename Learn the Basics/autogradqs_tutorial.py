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

自动微分 ``torch.autograd``
=======================================

在训练神经网络时，最常使用的算法是反向传播算法。在这种算法中，参数（模型权重）根据损失函数对参数的梯度来调整。

为了计算这些梯度，PyTorch有一个内置的微分引擎，叫做torch.autograd。它支持对任何计算图的梯度进行自动计算。

考虑最简单的单层神经网络，其输入为 ``x`` 、
参数 ``w ``和 ``b``，以及一些损失函数。它可以在
PyTorch中以如下方式定义:
"""

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


######################################################################
# 张量、函数和计算图
# ------------------------------------------
#
# 这段代码定义了以下 **计算图**:
#
# .. 图像特征:: /_static/img/basics/comp-graph.png
#    :alt:
#
# 在这个网络中，``w`` 和 ``b`` 是参数，我们需要进行
# 优化。因此，我们需要能够计算损失的梯度
# 函数与这些变量的关系。为了做到这一点，我们设置
# 这些张量的  ``requires_grad`` 属性。

#######################################################################
# .. note:: 你可以设置 ``requires_grad`` 的值在创造一个
#           张量时，或者之后通过使用 ``x.requires_grad_(True)`` 方法

#######################################################################
# 我们应用张量来构建计算图的函数，
# 实际上是 ``Function`` 类的对一个对象。 这个对象知道如何
# 在*向前*方向上计算该函数， 并且知道如何在
# 整个*反向传播*步骤中计算其导数. 对
# 后向传播函数的参考被存储在一个张量的 ``grad_fn`` 属性中。
#  你能找到更多关于 ``Function``的信息通过查阅
# 文档 <https://pytorch.org/docs/stable/autograd.html#function>`__.
#

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

######################################################################
# 计算梯度
# -------------------
#
# 在神经网络中为了优化参数的权重，我们需要
# 计算我们的损失函数对于参数的导数，
# 也就是说，我们需要求出:math:`\frac{\partial loss}{\partial w}` 和
# :math:`\frac{\partial loss}{\partial b}` 在
# ``x`` 和 ``y`` 的一些固定值下。为了计算导数， 我们称之为
# ``loss.backward()``， 然后再从``w.grad`` 和 ``b.grad`` 中获取数值 :
#

loss.backward()
print(w.grad)
print(b.grad)


######################################################################
# .. note::
#   - 我们只能获得计算图叶子节点的 ``grad`` 属性，
#     这些节点的 ``requires_grad`` 属性
#     设置为 ``True``。 对于计算图中的其他节点， 梯度将是
#     不可获取的。
#   - 我们只能使用
#     ``backward`` 在给定的图像上进行一次梯度计算，如果我们需要
#     在同一图像上做进行多次 ``backward`` 操作，我们需要设置
#     ``retain_graph=True`` 在 调用 ``backward`` 时。
#


######################################################################
# 禁用梯度跟踪
# ---------------------------
#
# 默认情况下，所有 ``requires_grad=True`` 的张量都在追踪其
# 计算历史并支持梯度计算。然而，在某些情况下
# 我们不需要这样做，例如，当我们已经
# 训练好模型，只是想把它应用于一些输入数据，也就是说，我们
# 只想通过网络进行*向前*计算。我们可以停止
# 追踪计算，方法是在我们的计算代码周围加上
# ``torch.no_grad()`` 块来停止跟踪计算：
#

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)


######################################################################
# 去实现相同结果的另一种方式是在张量上使用 ``detach()`` 方法
#

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

######################################################################
# 你可能想禁用梯度跟踪，这是有很多原因的：
# - 将神经网络中的某些参数标记为**冻结参数**。
# - 当你只做正向传递时，为了**加速计算**，因为对不跟踪梯度的张量计算
#   效率会更高。


######################################################################

######################################################################
# 关于计算图的更多信息
# ----------------------------
# 概念上，自动求导保留了数据（张量）的记录和所有执行的
# 操作（以及产生的新张量）的记录在一个有向无环
# 图（DAG）中，其中包括
# `Function <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>`__。
# 对象。在这个DAG中，叶子是输入的张量，根部是输出的
# 张量。通过追踪这个图从根到叶，你可以
# 使用链式规则自动计算梯度。
#
# 在前向传递时，autograd同时做两件事：
#
# - 运行所请求的操作，计算出一个结果张量。
# - 在DAG中保持该操作的*求导函数*。
#
# 当在DAG上调用 ``.backward()`` 时，反向传播开始。
# ``autograd`` 进行：
#
# - 从每个 ``.grad_fn`` 中计算梯度、
# - 将它们累积到各自的张量的 `.grad`` 属性中去
# - 使用链式规则，一直传播到叶子张量。
#

# .. note::
#   **DAGs在Pytorch中是动态变化的**
#   需要注意的是，图是从头开始重新创建的； 在每个
#   ``.backward()`` 调用后， autograd开始填充一个新图。这就是
#   允许你在你的模型中使用控制流语句的原因；
#   如果需要你可以在每次迭代时改变形状、大小和操作。

######################################################################
# 选择阅读： 张量梯度和雅各布乘积
# --------------------------------------
#
# 在许多情况下，我们有一个标量损失函数，并且我们需要计算
# 关于某些参数的梯度。然而，也有一些情况
# 当输出函数是一个任意的张量。在这种情况下，PyTorch
# 允许你计算所谓的**雅各布式乘积**，而不是计算实际的
# 梯度。
#
# 对于一个矢量函数 :math:`\vec{y}=f(\vec{x})`, 其中
# :math:`\vec{x}=\langle x_1,\dots,x_n\rangle` 和
# :math:`\vec{y}=\langle y_1,\dots,y_m\rangle`, 梯度
# :math:`\vec{y}` 相对于 :math:`\vec{x}` 是由**雅各布矩阵**给出的。
#
# .. math::
#
#
#    J=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)
#
# 与计算雅各布矩阵本身不同，PyTorch允许你计算
# **雅各布系数** :math:`v^T\cdot J`对于一个给定的输入矢量
# :math:`v=(v_1 \dots v_m)`。这是通过调用 ``backward`` 来实现的，
# :math:`v`作为一个参数。:math:`v`的大小应该与
# 原始张量的大小相同，相对于它来说，我们想要
# 计算乘积：
#

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")


######################################################################
# 注意，当我们第二次调用``backward``并使用相同的
# 参数时, 梯度的值是不同的. 发生这种情况是因为
# 在进行 `backward`` 传播时，PyTorch **累积
# 梯度**，也就是说，计算出的梯度值会添加到
# 计算图的所有叶子节点的 ``grad`` 属性。如果你想
# 计算适当的梯度，你需要在之前将 ``grad`` 属性清零。
# 在实际训练中，一个*优化器*可以帮助我们做到这一点。

######################################################################
# .. note:: 以前我们在调用 ``backward()`` 函数时没有
#           参数，这在本质上等同于调用 ``backward(torch.tensor(1.0))`` 在标量函数的情况下，这是计算梯度的有效方法，
#           例如在神经网络训练中的损失计算。
#

######################################################################
# --------------
#

#################################################################
# 更多阅读
# ~~~~~~~~~~~~~~~~~
# - `自动求导机制 <https://pytorch.org/docs/stable/notes/autograd.html>`_
