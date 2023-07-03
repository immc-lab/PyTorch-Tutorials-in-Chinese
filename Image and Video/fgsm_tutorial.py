# -*- coding: utf-8 -*-
"""
对抗性实例生成
==============================

**作者：** `Nathan Inkawhich <https://github.com/inkawhich>`__

如果你正在读这篇文章，希望你能体会到一些机器学习模型是多么有效。研究正在不断推动ML
模型变得更快、更准确、更高效。然而，设计和训练模型的一个经常被忽视的方面是安全性和
稳健性，特别是在面对那些不符合要求的模型时。

对ML模型的安全漏洞的认识，并对对抗性机器本教程将提高你学习这一热门话题进行深入了解。
你可能会惊讶地发现，在图像中添加不易察觉的扰动，*会导致模型性能的极大不同。考虑到这
是一个教程，我们将通过图像分类器的例子来探讨这个话题。具体来说，我们将使用一种最早和
最流行的攻击方法之一，快速梯度符号攻击（FGSM），来欺骗MNIST分类器。

"""


######################################################################
# 威胁模型
# ------------
# 
# 就背景而言，有许多类别的对抗性攻击，每一种都有不同的目标和对攻击者知识的假设。
# 然而，一般来说，首要的目标是向输入数据添加最少的扰动，以造成所需的错误分类。
# 攻击者的知识有几种假设，其中两种是：**白盒**和**黑盒**。**白盒**攻击假定
# 攻击者拥有对模型的全部知识和访问权，包括结构、输入、输出和权重。*黑盒*攻击假
# 定攻击者只能访问模型的输入和输出，而对底层结构或权重一无所知。还有几种类型的
# 目标，包括**错误分类**和**源/目标错误分类**。*错误分类*的目标意味着对手只希望
# 输出的分类是错误的，但并不关心新的分类是什么。*源/目标错误分类*意味着对手想改变
# 原本属于特定源类别的图像，使其被归类为特定目标类别。
# 
# 在这种情况下，FGSM攻击是一种*白盒*攻击，目标是*误分类*。
# 有了这些背景信息，我们现在可以详细讨论该攻击。
# 
# 快速梯度符号攻击
# -------------------------
# 
# 迄今为止，最早和最流行的对抗性攻击之一被称为*快速梯度符号攻击（FGSM）*，由Goodfellow等人在
# 《解释和利用对抗性实例》<https://arxiv.org/abs/1412.6572>`__中描述。该攻击是非常强大的，
# 但也是直观的。它旨在通过利用神经网络的学习方式（*梯度）来攻击神经网络。这个想法很简单，不是通过
# 基于反向传播梯度调整权重来最小化损失，而是攻击*调整输入数据以最大化损失*基于相同的反向传播梯度。
# 换句话说，攻击使用损失的梯度与输入数据的关系，然后 调整输入数据以使损失最大化。
# 
# 在我们进入代码之前，让我们看看著名的`FGSM <https://arxiv.org/abs/1412.6572>`__熊猫的
# 例子，并提取一些符号。
#
# .. figure:: /_static/img/fgsm_panda_image.png
#    :alt: fgsm_panda_image
#
# 从图中可以看出，:math:`\mathbf{x}`是被正确分类为 "熊猫 "的原始输入图像，:math:`y`是
# :math:`\mathbf{x}`的地面真实标签，:math:`\mathbf{\theta}`代表模型参数，
# :math:`J(\mathbf{x}, \mathbf{x}, y)`是用于训练网络的损失。攻击将梯度回传到输入数据，
# 以计算:math:`nabla_{x}J(\mathbf{theta}, \mathbf{x}, y)`。然后，它在方向上
# （即 :math:`sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))`）调整输入数据
# 一小步（:math:`epsilon`或 :math:`0.007`在图中），这将使损失最大化。
# 由此产生的扰动图像， :math:`x'`，然后被目标网络*错误分类*为 "长臂猿"，而实际上它仍然是 "熊猫"。
# 
# 希望现在本教程的动机是明确的，所以让我们进入实施阶段。
# 

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 小贴士: 这是一个黑客程序，用于绕过下载MNIST数据集时的 "用户代理 "限制，更多信息请参见，
#        https://github.com/pytorch/vision/issues/3497
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


######################################################################
# 实施
# --------------
# 
# 在本节中，我们将讨论本教程的输入参数、 定义被攻击的模型，然后对攻击进行编码并运行一些测试。
# 
# 输入
# ~~~~~~
# 
# 本教程只有三个输入，定义如下：
# 
# -  ``epsilons`` - 运行中使用的epsilon值的列表。在列表中保留0是很重要的，因为它代表了
#    模型在原始测试集上的表现。另外，从直觉上讲，我们希望epsilon越大，扰动越明显，但在降低
#    模型精度方面的攻击越有效。由于这里的数据范围是:math:`[0,1]`，所以epsilon值不应该超过1。
# 
# -  ``预训练的模型`` - 到预训练的MNIST模型的路径，该模型是用
#    `pytorch/examples/mnist <https://github.com/pytorch/examples/tree/master/mnist>`__训练的。
#    为了简单起见，在`这里 <https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing>`__下载预训练的模型。
# 
# -  ``使用cuda`` - 如果需要且可用，则用boolean标志来使用CUDA。
#    注意，带有CUDA的GPU对于本教程来说并不关键，因为CPU不会花费太多的时间。
# 

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True


######################################################################
# 受到攻击的模型
# ~~~~~~~~~~~~~~~~~~
# 
# 如前所述，被攻击的模型是来自`pytorch/examples/mnist
#  <https://github.com/pytorch/examples/tree/master/mnist>`__的同一个MNIST模型。
# 你可以训练并保存你自己的MNIST模型，或者你可以下载并使用提供的模型。这里的*Net*定义和测试数
# 据加载器都是从MNIST例子中复制的。
# 本节的目的是定义模型和数据加载器，然后初始化模型并加载预训练的权重。
# 

# LeNet模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST测试数据集和数据加载器声明
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])), 
        batch_size=1, shuffle=True)

# 定义我们使用的是什么设备
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 初始化网络
model = Net().to(device)

# 加载预训练的模型
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 将模型设置为评估模式。在这种情况下，这是针对Dropout层的。
model.eval()


######################################################################
# FGSM攻击
# ~~~~~~~~~~~
# 
# 现在，我们可以定义一个函数，通过扰动原始输入创建对抗性例子。`fgsm_attack'函数需要
# 三个输入，*image*是原始的干净图像（:math:`x`），*epsilon*是像素级的扰动量
# (:math:`\epsilon`)，*data_grad*是损失的梯度，与输入图像有关。
# (:math:`nabla_{x}) J(\mathbf{theta}, \mathbf{x}, y)`)。
# 然后，该函数创建扰动的图像为
# 
# .. math:: perturbed\_image = image + epsilon*sign(data\_grad) = x + \epsilon * sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))
# 
# 最后，为了保持数据的原始范围，扰动的图像被剪切到范围 :math:`[0,1]`。
# 

# FGSM攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的逐个元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动的图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪裁以保持[0,1]的范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回扰动后的图像
    return perturbed_image


######################################################################
# 测试函数
# ~~~~~~~~~~~~~~~~
# 
# 最后，本教程的核心结果来自于``test``函数。对这个测试函数的每次调用都会在MNIST测试集
# 上执行一个完整的测试步骤，并报告最终的准确性。然而，请注意，这个函数也需要一个*epsilon*
# 的输入。这是因为``test``函数报告了一个模型的准确度，该模型正受到具有:math:`epsilon`
# 强度的对手的攻击。更具体地说，对于测试集中的每个样本，该函数计算损失的梯度与输入数据
# (:math:`data\_grad`)的关系，用`fgsm_attack`(:math:`perturbed\_data`)创建一个
# 扰动的图像，然后检查扰动的例子是否是敌对的。除了测试模型的准确性外，该函数还保存并返回一些
# 成功的对抗性例子，以便以后进行可视化。
# 

def test( model, device, test_loader, epsilon ):

    # 准确度计数器
    correct = 0
    adv_examples = []

    # 在测试集的所有例子上循环
    for data, target in test_loader:

        # 发送数据和标签到设备
        data, target = data.to(device), target.to(device)

        # 设置张量的required_grad属性。对攻击很重要
        data.requires_grad = True

        # 通过模型向前传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # 得到最大对数概率的索引

        # 如果最初的预测是错误的，就不要再去攻击了，只要继续前进就行了
        if init_pred.item() != target.item():
            continue

        # 计算损失
        loss = F.nll_loss(output, target)

        # 将所有现有梯度归零
        model.zero_grad()

        # 在后向通道中计算模型的梯度
        loss.backward()

        # 收集 ``datagrad``
        data_grad = data.grad.data

        # 调用FGSM攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 对扰动的图像进行重新分类
        output = model(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1] # 获得最大对数概率的索引
        if final_pred.item() == target.item():
            correct += 1
            # 保存0 epsilon例子的特殊情况
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # 为以后的可视化保存一些adv的例子
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # 计算这个epsilon的最终精度
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确度和一个对抗性的例子
    return final_acc, adv_examples


######################################################################
# 运行攻击
# ~~~~~~~~~~
# 
# 实施的最后部分是实际运行攻击。在这里，我们为*epsilons*输入中的每个epsilon值运行一个
# 完整的测试步骤。对于每个epsilon，我们还保存了最终的准确度和一些成功的对抗性例子，在接
# 下来的章节中进行绘制。请注意，随着epsilon值的增加，打印出来的准确率是如何下降的？另外，
# 注意 :math:`epsilon=0`的情况代表原始的测试精度，没有攻击。
# 

accuracies = []
examples = []

# 对每个epsilon运行测试
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


######################################################################
# 结果
# -------
# 
# 准确度与Epsilon对比
# ~~~~~~~~~~~~~~~~~~~
# 
# 第一个结果是准确度与epsilon的关系图。正如前面提到的，随着epsilon的增加，我们期望测试
# 的准确性会下降。这是因为更大的epsilons意味着我们在将损失最大化的方向上迈出更大的一步。
# 注意，虽然epsilon值是线性间隔的，但是曲线的趋势不是线性的。例如，
# 在:math:`\epsilon=0.05`时的精度只比:math:`\epsilon=0`低4%，但在
# :math:`\epsilon=0.2`时的精度比:math:`\epsilon=0.15`低25%。另外，注意，模型的
# 准确度范围在:math:`\epsilon=0.25`和:math:`\epsilon=0.3`之间的10级分类器的随机准确度。
# 

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()


######################################################################
# 对抗性例子样本
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 还记得 "没有免费的午餐 "的观点吗？在这种情况下，随着epsilon的增加，测试精度下降，
# 但扰动变得更容易被感知。在现实中，攻击者必须考虑准确性下降和可感知性之间的权衡。这里，
# 我们展示了一些在每个epsilon值下成功的对抗性例子。图中的每一行都显示了不同的epsilon值。第
# 一行是 :math:`epsilon=0`的例子，代表没有扰动的原始 "干净 "图像。每张图片的标题显
# 示的是 "原始分类->对抗性分类"。注意，扰动在:math:`\epsilon=0.15`时开始变得明显，
# 在:math:`\epsilon=0.3`时相当明显。然而，在所有情况下，尽管增加了噪音，人类仍然能够
# 识别正确的类别。
# 

# 在每个epsilon下绘制几个对抗性样本的例子
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()


######################################################################
# 下一步怎么做？
# -----------------
# 
# 希望本教程能让大家对对抗性机器学习这一主题有一些了解。从这里开始，有很多潜在的方向可以研究。
# 这种攻击代表了对抗性攻击研究的开端，此后有很多关于如何从对抗者手中攻击和防御ML模型的想法。
# 事实上，在NIPS 2017上有一个对抗性攻击和防御比赛，比赛中使用的许多方法都在此文中描述：
# `对抗性攻击和防御比赛<https://arxiv.org/pdf/1804.00097.pdf>`__。防御方面的工作也
# 导致了使机器学习模型在一般情况下更加*坚固*的想法，对自然扰动和对抗性的输入都是如此。
# 
# 另一个方向是不同领域的对抗性攻击和防御。对抗性研究并不局限于图像领域，请看
# `这个<https://arxiv.org/pdf/1801.01944.pdf>`__对语音到文本模型的攻击。
# 但是，也许学习更多关于对抗性机器学习的最好方法是亲手去做。尝试实现NIPS 2017比
# 赛中的不同攻击，看看它与FGSM有什么不同。然后，尝试从你自己的攻击中捍卫该模型。
# 
