"""
**简介** ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动求导 <autogradyt_tutorial.html>`_ ||
`创建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
`理解模型 <captumyt.html>`_

PyTorch张量的介绍
===============================

请跟随下面的视频学习，或者你可以选择在 `youtube <https://www.youtube.com/watch?v=r7QDUPb2dCM>`__上观看此视频.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/r7QDUPb2dCM" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

张量是PyTorch的核心数据抽象。本节对 `torch.Tensor` 类进行了深入介绍。

首先，让我们导入PyTorch模块。我们还需导入Python的数学模块，以便一些例子的学习。

"""

import torch
import math


#########################################################################
# 创建张量
# ----------------
# 
# 调用 `torch.empty()` 是创建张量最简单的方式：
# 

x = torch.empty(3, 4)
print(type(x))
print(x)


##########################################################################
# 让我们将刚刚所做的分解一下：
# 
# -  我们使用 `torch` 模块的众多方法之一创建了一个张量。
# -  创建的是一个二维张量，它有三行四列。
# -  返回的对象的类型是torch.Tensor，它是torch.FloatTensor的别名；默认情况下，
#    PyTorch张量是32位浮点数类型的。(下面有更多关于数据类型的内容)。
# -  在打印你的张量时，你可能会看到一些随机的值。调用 `torch.empty()` 为张量分配了内存，
#    但并没有用任何值来初始化它--所以你所看到的是分配时内存中的存储的东西。
# 
# 关于张量及其维数的简要说明，以及术语：
# 
# -  你会看到一个一维的张量被称为*向量。*
# -  同样，一个二维的张量经常被称为*矩阵。*
# -  任何有两个维度以上的东西一般都被称为张量。
# 
# 更多的时候，你会想用一些值来初始化你的张量。常见的情况是所有的零、所有的一或随机值，
# 而 `torch` 模块为所有这些提供出厂方法：
# 

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)


#########################################################################
# 工厂方法做了你所期望的全部事情--我们有一个全零的张量，另一个是全一的张量，
# 还有一个是0和1之间的随机值的张量 。
# 
# 随机张量和种子
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 说到随机张量，你是否注意到紧接着对 `torch.manual_seed()` 的调用？用随机值初始化张量，
# 比如模型的学习权重，是很常见的，但是有些时候--特别是在研究环境中--你会希望对你的结果的可重复性有一些保证。
# 手动设置你的随机数发生器的种子就是这样做的。让我们详细地看看：
# 

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)


############################################################################
# 你在上面应该看到的是，`random1` 和 `random3` 带有相同的值，`random2` 和 `random4` 也是如此。
# 手动设置RNGa的种子会重置它，所以在大多数情况下，取决于随机数相同的计算应该提供相同的结果。
# 
# 更多详细信息, 请见 `PyTorch documentation on
# reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`__.
# 
# 张量的形状
# ~~~~~~~~~~~~~
# 
# 通常，当你对两个或多个张量进行操作时，它们需要具有相同的形状--即具有相同的维数和每个维数中相同的单元数。
# 为此，我们有 `torch.*_like()` 方法：
# 

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)


#########################################################################
# 上面代码单元中在张量上使用 `.shape` 属性。这个属性包含一个张量的每个维度的
# 范围列表--在我们的例子中，`x` 是一个三维张量，形状为2 x 2 x 3。
# 
# 在这下面，我们调用 `.empty_like()` 、`.zeros_like()` 、`.one_like()` 和 `.rand_like()` 方法。
# 使用 `.shape` 属性，我们可以验证这些方法中的每一个都返回一个维度和范围相同的张量。
# 
# 最后一种创建一个张量的方法是直接从PyTorch集合中指定其数据：
# 

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)


######################################################################
# 如果你已经有了Python元组或列表中的数据，创建张量最直接的方法是使用 `torch.tensor()` 。
# 如上所示，嵌套集合将产生一个多维张量。
# 
# .. 小贴士::
#      ``torch.tensor()`` 创建了数据的一份拷贝。
# 
# 张量数据类型
# ~~~~~~~~~~~~~~~~~
# 
# 设置张量的数据类型有几种方法：
# 

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)


##########################################################################
# 设置张量的底层数据类型的最简单方法是在创建时使用一个可选的参数。在上面单元格的第一行，
# 我们为张量 `a` 设置 `dtype=torch.int16`。当我们打印 `a` 时，
# 我们可以看到它的填充值为 `1`，而不是 `1.` --这是Python的一个巧妙的机制，
# 提示我们这是一个整数类型，而不是浮点类型。
# 
# 关于打印 `a`，需要注意的另一件事是，与我们将 `dtype` 作为默认值（32位浮点）不同，
# 打印张量时会指明其 `dtype`。
# 
# 你可能还发现，我们从一系列的整数参数指定张量的形状，到将这些参数分组为一个元组。
# 这并不是必须的--PyTorch会将一系列初始的、未标记的整数参数作为张量的形状--但在添加可选的参数时，
# 它可以使你的意图更容易阅读。
# 
# 设置数据类型的另一种方法是使用 `.to()` 方法。在上面的单元格中，我们创建一个随机的浮点张量 `b`。
# 之后，我们用 `.to()` 方法将b转换为32位整数，从而创建 `c` 。注意，`c` 包含所有与 `b` 相同的值，但被截断为整数。
# 
# 可用的数据类型包括：
# 
# -  ``torch.bool``
# -  ``torch.int8``
# -  ``torch.uint8``
# -  ``torch.int16``
# -  ``torch.int32``
# -  ``torch.int64``
# -  ``torch.half``
# -  ``torch.float``
# -  ``torch.double``
# -  ``torch.bfloat``
# 
# 使用PyTorch Tensors的数学与逻辑
# ---------------------------------
# 
# 现在，你知道了一些创建张量的方法...那么你可以对他做什么呢？
# 
# 让我们先看看基本的算术，以及张量如何与简单的标量进行运算：
# 

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)


##########################################################################
# 正如你在上面看到的，张量和标量之间的算术运算，如加法、减法、乘法、除法和
# 指数化都分布在张量的每个元素上。因为这种操作的输出将是一个张量，
# 你可以用通常的运算符优先规则将它们链接起来，就像我们创建 `threes` 的那一行。
# 
# 两个张量之间的类似操作也是如此：
# 

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)


##########################################################################
# 这里需要注意的是，前面的代码单元中所有的张量都是相同的形状。当我们试图对形状不同的张量
# 进行二元运算时会发生什么？形状不同的张量进行二元操作会发生什么？
# 
# .. 小贴士::
#      下面的单元格会抛出一个运行时错误。这是有意的。
#
# ::
#
#    a = torch.rand(2, 3)
#    b = torch.rand(3, 2)
#
#    print(a * b)
#


##########################################################################
# 在一般情况下，你不能以这种方式对不同形状的张量进行操作，即使是在像上面的单元格那样的情况下，
# 张量的元素数量相同，也不能对不同形状的张量进行操作。
# 
# 简要介绍一下： 张量广播
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# .. 小贴士::
#      如果你熟悉NumPy ndarrays中的广播机制，你会发现这里也适用同样的规则。
# 
# 与相同形状规则所不同的是张量的广播。请看下面的例子：
# 

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)


#########################################################################
# 这里有什么技巧？我们是如何让2x4的张量和1x4的张量相乘的？
# 
# 广播是在形状相似的张量之间进行操作的一种方式。在上面的例子中，1行4列的张量与2行4列的张量的两行都进行了相乘。
# 
# 这是深度学习中的一个重要操作。常见的情况是将学习权重的张量与一批输入张量相乘，对该批中的每个实例分别应用该操作，
# 并返回一个形状相同的张量--就像我们上面的 (2，4) * (1，4) 的例子一样，返回一个形状为（2，4）的张量。
# 
# 广播的规则是：
# 
# -  每个张量至少有一维，不能是空张量。
# 
# -  从尾到头比较两个张量的维度大小:
# 
#    -  每个维度必须相等， *或*
# 
#    -  一个维度必须为1， *或*
# 
#    -  维度不存在于其中一个张量
# 
# 当然，形状相同的张量一定 "可广播"，正如你前面所看到的。
# 
# 下面是一些遵守上述规则并允许广播的情况的例子：
# 

a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 第三和第二维与a相同，没有第一维
print(b)

c = a * torch.rand(   3, 1) # 第三维为1， 第二维与a相同
print(c)

d = a * torch.rand(   1, 2) # 第三维与a相同，第二维为1
print(d)


#############################################################################
# 仔细看上面的每一个张量的值：
#
# -  创建 `b` 的乘法运算是在 `a` 的每一"层"上进行的。
# -  对于 `c`，该操作在 `a` 的每一层和每一行都被广播了--每一个3元素的列都是相同的。
# -  对于 `d`，我们把它换了一下--现在每一行都是相同的，跨越层和列。
# 
# 有关广播的更多信息，请参见 `PyTorch
# 的相关文档 <https://pytorch.org/docs/stable/notes/broadcasting.html>`__。
# 
# 下面是一些尝试广播会失败的例子：
# 
# .. 小贴士::
#       下面的每个单元格都会抛出 run-time error。这么做是有意的。
#
# ::
#
#    a =     torch.ones(4, 3, 2)
#
#    b = a * torch.rand(4, 3)    # 维度必须从尾到头匹配
#
#    c = a * torch.rand(   2, 3) # 第二和第三维均不相同
#
#    d = a * torch.rand((0, ))   # 空张量不能广播
#


###########################################################################
# 更多关于张量的数学知识
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# PyTorch张量有超过三百种可以对其进行的运算。
# 
# 下面是一些主要的运算中的例子：
# 

# 一般函数
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# 三角函数和它们的反函数
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# 比特运算
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# 比较：
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # 多比较操作都支持广播！
print(torch.eq(d, e)) # 返回一个布尔型的张量

# 消减：
print('\nReduction ops:')
print(torch.max(d))        # 返回一个单值张量
print(torch.max(d).item()) # 从返回的张量抽取数值
print(torch.mean(d))       # 求张量均值
print(torch.std(d))        # 标准差
print(torch.prod(d))       # 所有数字相乘
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # 过滤唯一元素

# 向量和线性代数运算
v1 = torch.tensor([1., 0., 0.])         # x单位向量
v2 = torch.tensor([0., 1., 0.])         # y单位向量
m1 = torch.rand(2, 2)                   # 随机数矩阵
m2 = torch.tensor([[3., 0.], [0., 3.]]) # 3倍特征矩阵

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # z单位向量的负(v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 3乘m1
print(torch.svd(m3))       # 奇异值分解


##################################################################################
# 这只是一个小的运算样例。更多的细节和完整的数学函数清单，请看
# `文档 <https://pytorch.org/docs/stable/torch.html#math-operations>`__.
# 
# 就地改变张量
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 大多数对张量的二元运算将返回第三个新张量。当我们有 `c = a * b`（其中 `a` 和 `b` 是张量），
# 新的张量 `c` 将开辟一个与其他张量不同的内存区域。
# 
# 不过，有些时候你可能希望就地改变张量--例如，如果你正在做一个可以丢弃中间值的元素小计算。
# 为此，大多数数学函数都有一个带有下划线（`_`）的版本，可以就地改变一个张量。
# 
# 例如:
# 

a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # 这个运算在内存中创建了一个行的张量
print(a)              # a本身并未改变

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # 注意下划线
print(b)              # b已经改变了


#######################################################################
# 对于算数运算，也是一样的：
# 

a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)


##########################################################################
# 请注意，这些就地算术函数是 `torch.Tensor` 对象上的方法，而不是像许多其他函数
# （如`torch.sin()` ）那样附属于 `torch` 模块。从 `a.add_(b)` 可以看出，调用的张量是被就地改变的。
# 
# 有一个选择是将计算的结果放在一个现有的、分配的张量中。到目前为止，我们所看到的许多方法和函数--包括创建方法！
# 都有一个 `out` 参数，它可以让你指定一个张量来接收输出！- 有一个out参数，让你指定一个张量来接收输出。
# 如果输出张量是正确的形状和 `dtype`，这不会进行新的内存分配：
# 

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # c中的值已经改变了

assert c is d           # 测试c和d是同一个对象，不仅仅比较它们的值
assert id(c) == old_id  # 确保新的c和旧的是同一个对象

torch.rand(2, 2, out=c) # 对于创建也同样生效！
print(c)                # c又一次改变了
assert id(c) == old_id  # 仍然是同一个对象！


##########################################################################
# 复制张量
# ---------------
# 
# 和Python中的对象一样，将张量赋值给一个变量会使该变量成为张量的*标签*，而不是复制它。比如说:
# 

a = torch.ones(2, 2)
b = a

a[0][1] = 561  # 我们改变a
print(b)       # b的值也发生了改变


######################################################################
# 但是如果你想要一个单独的数据副本来工作呢？`clone()`方法就是为你而设的：
# 

a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # 在内存中的不同对象
print(torch.eq(a, b))  # 但是值依旧相等！

a[0][1] = 561          # a变了
print(b)               # 但是b没改变


#########################################################################
# **当调用在使用 `clone()` 时，有一件重要的事情需要注意**。如果你的源张量启用了autograd，
# 那么克隆的也会启用。**这将在关于autograd的视频中更深入地介绍**，但如果你想了解更多的细节，请继续。
# 
# 在许多情况下，你需要这么做。例如，如果你的模型在其forward()方法中有多个计算路径，
# 并且原始张量和其克隆对模型的输出都有贡献，那么为了实现模型学习，你希望对两个张量都打开autograd。
# 如果你的源张量已经启用了autograd（如果它是一组学习权重或者是从涉及权重的计算中派生出来的，
# 那么它通常会启用），那么你会得到你想要的结果。
# 
# 另一方面，如果你在做一个计算，原始张量和它的克隆都不需要跟踪梯度，那么只要关闭源张量的autograd就可以了。
# 
# 不过还有第三种情况：想象一下你在模型的 `forward()` 函数中进行计算，梯度在默认情况下是打开的，
# 但是你想在中途抽出一些值来生成一些指标。在这种情况下，你不希望你的源张量的克隆副本
# 跟踪梯度--关闭autograd的历史跟踪后，性能会得到改善。为此，你可以在源张量上使用`.detach()` 方法：
# 

a = torch.rand(2, 2, requires_grad=True) # 打开autograd
print(a)
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)


#########################################################################
# 这里发生了什么？
# 
# -  我们创建了 `a` 并将它的 `requires_grad=True` 打开。
#    **我们所讲的目前还没涉及到这一操作，但是在autogard单元中会涉及到。**
# -  当我们打印 `a`，它告诉我们属性 `requires_grad=True`，这意味着autogard和计算历史追踪已经打开。
# -  我们克隆 `a` 并且用 `b` 表示。当我们打印 `b`，我们可以看到它正在追踪它的计算历史，
#    它继承了 `a` 的autograd，并且加入了计算历史。
# -  我们把 `a` 克隆到 `c`，但是我们首先调用了 `detach()`。
# -  打印 `c`，我们没有看到计算历史，也没看到 `requires_grad=True`。
# 
# `detach()` 方法将张量从其计算历史中分离出来。它说，"接下来的事情都是在关闭autograd的情况下进行的"。
# 它是在不改变 `a` 的情况下进行的, 你可以看到，当我们在最后再次打印 `a` 时，它保留了 `requires_grad=True` 的属性。
# 
# 转移到GPU
# -------------
# 
# PyTorch的主要优势之一在于它在兼容CUDA的Nvidia GPU上的强大加速能力。("CUDA "是Compute Unified Device Architecture的缩写，
# 它是Nvidia的并行计算平台。) 到目前为止，我们所做的一切都是在CPU上进行的。我们怎样才能将这些操作转移到更快的硬件上呢？
# 
# 首先，我们应该通过 `is_available()` 方法检查GPU是否可用。
# 
# .. 小贴士::
#      如果你没有安装兼容CUDA的GPU和CUDA驱动，本节中的可执行单元将不会执行任何 GPU相关的代码。
# 

if torch.cuda.is_available():
    print('我们有GPU!')
else:
    print('很遗憾, 你只有CPU。')


##########################################################################
# 一但我们确定了一个或多个GPU可用，我们就需要将我们的数据放到GPU能看到的地方。你的CPU会在你电脑上的RAM上进行运算。
# 你的GPU有专用的内存连接到它。每当你想在一个设备上进行计算时，你必须将该计算所需的所有数据转移到该设备可访问的内存中。
# (俗话说，"把数据移到GPU可访问的内存 "被简称为 "把数据移到GPU")。
# 
# 有多种方法可以将你的数据带到你的目标设备上。你可以在创建时进行：
# 

if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('很遗憾，你只有CPU。')


##########################################################################
# 默认情况下，新的张量会被创建在CPU上，所以当我们想将张量创建到GPU上时，我们需要指定可选参数 `device`。
# 当我们打印新的张量时，PyTorch会告诉我们它在哪个设备上（如果不在CPU上）。
# 
# 你可以用 `torch.cuda.device_count()` 来查询GPU的数量。如果你的GPU数量大于1，
# 你可以通过指定下标来指定某一个：`device='cuda:0'`, `device='cuda:1'`,等等。
# 
# 作为一种编码实践，用字符串常量来指定我们的设备是不好的。在一个理想情况下，无论你是在CPU还是GPU硬件上，
# 你的代码都应该表现得很棒。你可以通过创建一个可以传递给你的张量的设备句柄，而不是一个字符串来实现这一点：
# 

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)


#########################################################################
# 如果你已经有了一个在某一设备上的张量，你可以使用 `to()` 方法来将它移动到另一个设备。
# 下面的代码在CPU上创建了一个张量，并把它移到你在前一个单元格中获得的设备句柄上。
# 

y = torch.rand(2, 2)
y = y.to(my_device)


##########################################################################
# 重要的是要知道，为了进行涉及两个或更多张量的计算，所有的张量必须在同一个设备上。
# 无论你是否有一个可用的GPU设备，下面的代码都会引发一个运行时错误：
# 
# ::
# 
#    x = torch.rand(2, 2)
#    y = torch.rand(2, 2, device='gpu')
#    z = x + y  # 将会抛出异常
# 


###########################################################################
# 操纵张量的形状
# --------------------------
# 
# 有时，你需要改变你的张量的形状。下面，我们来看看几种常见的情况，以及如何处理它们。
# 
# 改变尺寸的数量
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 你可能需要改变维数的一种情况是将单个输入实例传递给你的模型。PyTorch模型通常希望有成批的输入。
# 
# 例如，有一个在3 x 226 x 226图像上工作的模型--一个226像素的正方形，有3个颜色通道。当你加载和转换它时，
# 你会得到一个形状的张量 `(3, 226, 226)`。但是，你的模型希望输入的是形状 `(N, 3, 226, 226)`，
# 其中N是一个批次中的图像数量。那么，你如何制作一个批次的图像呢？
# 

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0) # 在0维扩充一个为 1 的维度

print(a.shape)
print(b.shape)


##########################################################################
# `unsqueeze()` 方法添加了一个范围为1的维度，
# `unsqueeze(0)` 将其作为新的第0个维度添加--现在你有了一个批量为1的维度!
# 
# 那么，如果这就是unsqueezing？我们所说的squeezing是什么意思？
# 我们正在利用这样一个事实：任何范围为1的维度都不会改变张量中的元素数量。
# 

c = torch.rand(1, 1, 1, 1, 1)
print(c)


##########################################################################
# 继续上面的例子，假设模型对每个输入的输出是20个元素的向量。那么你会期望输出的形状是（N，20），
# 其中N是输入批次中的实例数。这意味着，对于我们的单一输入批次，我们将得到一个形状为（1，20）的输出。
# 
# 如果你想用这个输出做一些非批处理的计算，只是想得到一个20元素的向量，该怎么做？
# 

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)


#########################################################################
# 你可以从shape中看到，我们的二维张量现在是一维的，如果你仔细观察上面单元格的输出，
# 你会发现打印出的 `a` 显示了一组 “额外的” 方括号 `[]`，因为有一个额外的维度。
# 
# 你只能 `squeezing()` 大小为1的维度。请看上面，我们试图在 `c` 中 `squeezing` 一个2的维度，结果得到的是与开始相同的形状。
# 对 `squeeze()` 和 `unsqueeze()` 的调用只能作用于范围为1的维度，否则会改变张量的元素数量。
# 
# 你可能使用 `unsqueeze()` 的另一个地方是为了缓解广播。回顾上面的例子，我们有以下代码：
# 
# ::
# 
#    a =     torch.ones(4, 3, 2)
# 
#    c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
#    print(c)
# 
# 这样做的效果是在维度0和维度2上进行广播操作，导致随机的3×1张量被 `a` 中的每一个3元素的列进行元素相乘。
# 
# 如果随机向量只是3元素的向量呢？我们就会失去做广播的能力，因为根据广播规则，最终的尺寸不会匹配。
# `unsqueeze()` 会来拯救我们：
# 

a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # 尝试 a * b 将会返回一个运行时错误
c = b.unsqueeze(1)       # 改变一个二维张量，在末尾新加一个维度
print(c.shape)
print(a * c)             # 广播再次生效了！


######################################################################
# `squeeze()`和`unsqueeze()` 方法都有一个原地操作版本，`squeeze_()` 和 `unsqueeze_()`:
# 

batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)


##########################################################################
# 有时你会想更彻底地改变张量的形状，同时仍然保留元素的数量和它们的内容。一种情况是在模型的卷积层和
# 模型的线性层之间的层--这在图像分类模型中很常见。卷积核会产生一个形状x宽度x高度的输出张量，
# 但是下面的线性层希望得到一个一维的输入。`reshape()` 会帮你做这个，
# 只要你要求的维度产生的元素数量与输入张量相同：
# 

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# 也可以把它作为torch模块的一个方法来调用:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)


###############################################################################
# .. 小贴士::
#      上面单元格最后一行的`(6 * 20 * 20,)`参数是因为PyTorch在指定张量形状时希望输入是一个**元组**，
#      但当形状是方法的第一个参数时，它允许我们偷懒，只使用一系列的整数。在这里，我们不得不添加括号和逗号，
#      以说服方法，这实际上是一个单元素元组。
# 
# `reshape()`将返回一个要改变的张量上的*视图*，也就是说，一个单独的张量对象看同一个内存底层区域。
# 这一点很重要：这意味着对源张量的任何改变都会反映在该张量的视图中，除非你`clone()` 它。
# 
# 有些情况，超出本介绍的范围，reshape()必须返回一个携带数据副本的张量。欲了解更多信息，请参见
# `文档 <https://pytorch.org/docs/stable/torch.html#torch.reshape>`__。
# 


#######################################################################
# NumPy桥
# ------------
# 
# 在上面关于广播的章节中提到，PyTorch的广播与NumPy是兼容的--但PyTorch和NumPy之间的关系甚至比这更深。
# 
# 如果你现有的ML或科学代码中的数据存储在NumPy的ndarrays中，你可能希望将相同的数据转换为PyTorch的张量，
# 无论是利用PyTorch的GPU加速，还是为建立机器学习模型。在ndarrays和PyTorch tensors之间切换很容易：
# 

import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)


##########################################################################
# PyTorch创建了一个与NumPy数组相同形状，相同数据的张量，保留的数据类型为NumPy默认的64位浮点数类型。
# 
# 反向的转换也很容易：
# 

pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)


##########################################################################
# 重要的是要知道，这些转换后的对象与它们的源对象使用相同的底层内存，
# 这意味着一个对象的变化会反映在另一个对象上：
# 

numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
