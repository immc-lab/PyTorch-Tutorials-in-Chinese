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

张量
==========================

张量是一种特定的数据结构，与数组和矩阵非常相似。在PyTorch中，我们使用张量来编码一个模型的输入和输出，以及模型的参数。

张量与`NumPy<https://numpy.org/>`_的ndarrays类似，只是张量可以在GPU或其他硬件加速器上运行。事实上，张量和NumPy数组通常可以共享相同的底层内存，
不需要复制数据（见 :ref:`bridge-to-np-label`）。张量还为自动求导进行了优化（我们将在后面的`自动微分 <autogradqs_tutorial.html>`__部分看到更多关于这一点）。
如果你熟悉ndarrays，你就会对张量API的很熟悉。如果不熟悉，请跟上!
"""

import torch
import numpy as np


######################################################################
# 初始化张量
# ~~~~~~~~~~~~~~~~~~~~~
#
# 张量可以用不同的方法进行初始化。请看下面的例子：
#
# **直接从数据中创建**
#
# 张量可以直接使用数据创建。数据类型会自动匹配。

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

######################################################################
# **从NumPy数组中创建**
#
# 张量可以用NumPy数组创建 (反之亦然，详见 :ref:`bridge-to-np-label`).
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


###############################################################
# **用其他张量创建:**
#
# 除非明确的重写，新的张量会保留参数张量的属性（形状、数据类型）。

x_ones = torch.ones_like(x_data) # 创建了一个全1的张量，新张量保留了x_data的属性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 创建了一个随机数张量，重写了数据类型属性
print(f"Random Tensor: \n {x_rand} \n")


######################################################################
# **用随机值或常数值创建:**
#
# ``shape`` 是一个描述张量维度的元组。在下面的函数中，它定义了输出张量的维度。

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



######################################################################
# --------------
#

######################################################################
# 张量的属性
# ~~~~~~~~~~~~~~~~~
#
# 张量的属性描述了他们的形状、数据类型和它们被存储在哪个设备上。

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


######################################################################
# --------------
#

######################################################################
# 张量上的操作
# ~~~~~~~~~~~~~~~~~
#
# `这里 <https://pytorch.org/docs/stable/torch.html>`__全面介绍了100多个
# 张量操作，包括算术、线性代数、矩阵操作（转置、索引、切片）、采样等。
#
# 这些操作中的每一个都可以在GPU上运行（速度通常比在CPU上高）。如果你使用Colab，可以
# 进入Runtime > Change runtime type > GPU来分配一个GPU。
#
# 默认情况下，张量是在CPU上创建的。我们需要使用 ``.to`` 方法显式地将张量移动到GPU上
# （在检查GPU的可用性之后）。请注意，在不同的设备上复制大型的张量，在时间和内存上都是很昂贵的。

# 如果GPU可用，我们就将张量移动到GPU上。
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


######################################################################
# 尝试一下列表（list）中的一些操作。
# 如果你熟悉NumPy API，你会发现Tensor API使用起来很容易。
#

###############################################################
# **类似NumPy的索引和切片:**

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

######################################################################
# **拼接张量** 你可以使用 ``torch.cat`` 来沿着给定的维度拼接一组张量。
# 也请参见 `torch.stack <https://pytorch.org/docs/stable/generated/torch.stack.html>`__,
# 另一个与 ``torch.cat`` 有细微差别的张量连接运算符。
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


######################################################################
# **算术运算**

# 两个张量的矩阵乘法。y1, y2, y3具有相同的值。
# ``tensor.T`` 返回一个张量的转置。
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# 张量之间对应元素相乘。z1, z2, z3具有相同的值。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


######################################################################
# **单元素张量** 如果你有一个单元素张量，例如通过将一个张量的所有值聚集成一个值，
# 你可以使用 ``item()``将其转换为一个 Python 数值：

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


######################################################################
# **原地操作**
# 将结果存储到操作数中的操作被称为原地操作。它们用后缀 ``_`` 来表示。
# 例如： ``x.copy_(y)``, ``x.t_()``, 这样的操作将改变 ``x``本身。

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

######################################################################
# .. 小贴士::
#      原地操作可以节省一些内存，但在计算导数时可能会出现问题，因为会丢失历史记录。
#      因此，我们不鼓励使用这种操作。



######################################################################
# --------------
#


######################################################################
# .. _bridge-to-np-label:
#
# 与NumPy桥接
# ~~~~~~~~~~~~~~~~~
# CPU上的张量和NumPy数组可以共享它们的底层内存位置，所以改变一个将改变另一个。


######################################################################
# 张量转NumPy数组
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

######################################################################
# 对tensor进行修改，将会反映到NumPy数组上。

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


######################################################################
# NumPy数组转张量
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n = np.ones(5)
t = torch.from_numpy(n)

######################################################################
# 改变NumPy数组会反映到张量上。
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
