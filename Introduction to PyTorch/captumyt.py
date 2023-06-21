"""
`简介 <introyt1_tutorial.html>`_ ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动求导 <autogradyt_tutorial.html>`_ ||
`创建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
**理解模型**

用Captum理解模型
===============================

请跟随下面的视频学习，或者你可以选择在 `youtube <https://www.youtube.com/watch?v=Am2EF9CLu-g>`__上观看此视频。
在`这里 <https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip>`__下载笔记和相应的文件。

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Am2EF9CLu-g" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

`Captum <https://captum.ai/>`__ (在拉丁文中是“理解力”的意思) 是一个开源的，建立在Pytorch基础上
可扩展的模型解释库。
随着模型复杂性的增加和由此产生的缺乏透明度，模型的解释方法变得越来越重要。模型理解既是一个活跃的研究领域，
也是一个使用机器学习的跨行业实际应用的重点领域。 Captum提供了最先进的算法，包括综合梯度，这些算法为研究人员
和开发人员提供了一种简单的方法去了解哪些特征对模型的输出有贡献。

完整的文档、API参考和一套关于特定主题的教程可在以下网站获得 `captum.ai <https://captum.ai/>`__。

简介
------------

Captum对模型解释的处理方法是：*归因。*
在Captum中，有三种归因法可供选择：

-  **特征归因** 以输入生成的特征来解释特定的输出。以影评中的某些词为依据，
   来判断一个影评是正面的还是负面的，这就是特征归因的一个例子。
-  **层归因** 检查模型的隐藏层在特定输入后的活动。 为了响应输入图像，
   检查卷积层的空间映射输出，这是一个层归因的例子。
-  **神经元归因** 类似于层归因，但侧重于单个神经元的活动。

在这个交互式笔记中，我们来看一下特征归因和层归因。

在这三种归因类型中，每一种归因都有多种 **归因算法** 相关联。
许多归因算法分为两大类：

-  **基于梯度的算法** 计算模型输出、层输出或神经元激活相对于输入的后向梯度。
   **综合梯度** (针对特征)， **层梯度激活**， 和 **神经元电导率** 都是基于梯度的算法。
-  **基于扰动的算法** 检查了在输入变化的反应中，模型、层或神经元的输出的变化。输入扰动
   可以是定向的或随机的。 **遮挡，** **特征消融，** 和 **特征排列组合** 都是基于扰动的算法。

我们将在下面研究这两种类型的算法。

特别是在涉及大模型的情况下，将归因数据可视化，使其与正在检查的输入特征轻松联系起来，可能是很有价值的。
虽然当然可以用Matplotlib、Plotly或类似的工具来创建自己的可视化，但Captum提供了专门针对其属性的增强工具：

-   ``captum.attr.visualization`` 模块 (下面以 ``viz`` 的形式导入)
   提供了与图像相关的属性可视化的有用功能。
-  **Captum Insights** 是在Captum之上的一个易于使用的API，它提供了一个可视化的小部件，有现成的图像、
   文本和任意模型类型的可视化。

这两个可视化工具集都将在本笔记本中演示。前面的几个例子将集中在计算机视觉的用例上，但最后的Captum Insights
部分将展示多模型、可视化问答模型中的归因的可视化。

安装
------------

在你开始之前，你需要有一个Python环境：

-  Python 3.6或更高版本
-  对于Captum Insights的例子，Flask 1.1或更高版本
-  PyTorch 1.2或更高版本（建议使用最新版本）。
-  TorchVision 0.6或更高版本（建议使用最新版本）。
-  Captum（建议使用最新版本）。

要在Anaconda或pip虚拟环境中安装Captum，请使用以下适合您的环境的命令：

用 ``conda``::

    conda install pytorch torchvision captum -c pytorch

用 ``pip``::

    pip install torch torchvision captum

在你设置的环境中重新启动这个笔记，你就可以开始工作了!


第一个例子
---------------
 
首先，我们举一个简单、直观的例子。我们从一个在ImageNet数据集上预训练的ResNet模型开始。
我们将得到一个测试输入，并使用不同的 **特征归因** 算法来检查输入图片如何影响输出。同时，
我们还会看到一些测试图片的输入归因图的可视化。

 
首先，导入模块:

"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#########################################################################
# 现在我们将使用TorchVision模型库来下载一个预训练的ResNet。因为我们不是在训练，所以我们
# 将暂时把它放在评估模式下。
# 

model = models.resnet101(weights='IMAGENET1K_V1')
model = model.eval()


#######################################################################
# 你得到这个交互式笔记本的地方也应该有一个 ``img`` 文件夹，这个文件夹里面有一个 ``cat.jpg`` 文件。
# 

test_img = Image.open('img/cat.jpg')
test_img_data = np.asarray(test_img)
plt.imshow(test_img_data)
plt.show()


##########################################################################
# 我们的ResNet模型是在ImageNet数据集上训练出来的，模型要求图像有确定的大小，
# 并且通道数据应规范到特定的数值范围。 我们还应为模型识别的类别拉入人类可读的
# 标签列表--这也应该在 ``img`` 文件夹中。
# 

# 模型要求224x224的3色图像
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

# 标准的ImageNet规范化
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0) # 模型需要一个虚拟的批次维度

labels_path = 'img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


######################################################################
# 现在，我们提出一个问题： 我们的模型认为这个图像代表什么？
# 

output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


######################################################################
# 我们已经确认，ResNet认为我们的猫的图像实际上是一只猫。但 *为什么* 模型认为这是一只猫的图像呢？
# 
# 对于这个问题的答案，我们转向Captum。
# 


##########################################################################
# 具有综合梯度的特征归因
# ---------------------------------------------
# 
# **特征归因** 将一个特定的输出归于输入的特征。它用一个特定的输入--这里指我们的测试图像--来
# 生成每个输入特征对特定输出特征的相对重要的图。
# 
# `综合
# 梯度 <https://captum.ai/api/integrated_gradients.html>`__ 是Captum中可用的
# 特征归属算法之一。综合梯度通过逼近模型输出相对于输入的梯度的积分，给每个输入特征分配一个重
# 要的分数。
# 
# 在我们的案例中，我们将采取输出向量的一个特定元素--即表示模型对其所选类别的信心的元素--并使
# 用综合梯度来了解输入图像的哪些部分促成了这一输出。
# 
# 我们有了综合梯度的重要性图之后，我们将使用Captum中的可视化工具来表示重要性图。Captum的
# ``visualize_image_attr()`` 函数提供了多种选项来定制属性数据的显示。
# 在这里，我们传入一个自定义的Matplotlib颜色图。
# 
# 调用 ``integrated_gradients.attribute()`` 运行单元格，通常需要一两分钟。
# 

# 用模型初始化归因算法
integrated_gradients = IntegratedGradients(model)

# 要求该算法归结我们的输出目标
attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

# 显示原始图像进行比较
_ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)), 
                      method="original_image", title="Original Image")

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             title='Integrated Gradients')


#######################################################################
# 在上图中，你应该看到，综合梯度给我们提供了图像中猫的位置周围最强的信号。
# 


##########################################################################
# 有遮挡的特征归因
# ----------------------------------
# 
# 基于梯度的归因方法有助于理解模型，即直接计算出相对于输入的输出变化。
# *基于扰动的归因* 方法更直接地处理这个问题，通过引入输入的变化来测量对输出的影响。
# `遮挡 <https://captum.ai/api/occlusion.html>`__ 就是这样一种方法。
# 遮挡涉及替换输入图像的部分，并检查对输出信号的影响。
# 
# 下面，我们设置了Occlusion属性。与配置卷积神经网络类似，你可以指定目标区域的大小，
# 以及一个跨度长度来确定各个测量的间隔。我们将用 ``visualize_image_attr_multiple()``
# 来可视化我们的Occlusion归因的输出，按区域显示正面和负面归因的热力图，并通过用正面归因的区域
# 掩盖原始图像。遮蔽可以提供一个非常有启发性的观点，即模型认为我们的猫咪照片中哪些区域最 "像猫"。
# 

occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input_img,
                                       target=pred_label_idx,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)


_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                      fig_size=(18, 6)
                                     )


######################################################################
# 我们再次看到，图像中包含猫的区域置于更重要的位置。
# 


#########################################################################
# 用层GradCAM进行层归因
# ------------------------------------
# 
# **层归因** 允许你将模型中隐藏层的活动归因于输入的特征。
# 下面，我们将使用一个层归属算法来检查我们模型中的一个卷积层的活动。
# 
# GradCAM计算目标输出相对于给定层的梯度，对每个输出通道（输出的维度2）进行平均，并将每个通道的
# 平均梯度乘以层的激活。结果是所有通道的总和。GradCAM是为卷积网设计的；由于卷积层的活动经常在空
# 间上映射到输入，所以GradCAM的属性经常被放大并用于掩盖输入。
# 
# 层归因的设置与输入归因类似，只是除了模型之外，你必须在模型中指定一个你想检查的隐藏层。如上所述，
# 当我们调用 ``attribute()``时，我们指定感兴趣的目标类。
# 

layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                             sign="all",
                             title="Layer 3 Block 1 Conv 2")


##########################################################################
# 我们将使用
# `层归因 <https://captum.ai/api/base_classes.html?highlight=layerattribution#captum.attr.LayerAttribution>`__
# 基类中的方便方法 ``interpolate()`` 来对这个属性数据进行放大，以便与输入图像进行比较。
# 

upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

_ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      transformed_img.permute(1,2,0).numpy(),
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      fig_size=(18, 6))


#######################################################################
# 像这样的可视化可以让你对你的隐藏层如何响应你的输入有新的认识。
# 


##########################################################################
# 用Captum Insights进行可视化
# ----------------------------------
# 
# Captum Insights是一个建立在Captum之上的可解释的可视化部件，用于促进模型的理解。
# Captum Insights跨越图像、文本和其他特征，帮助用户理解特征归属。它允许你对多个输
# 入/输出对进行可视化归因，并为图像、文本和任意数据提供可视化工具。
# 
# 在笔记的这一部分，我们将用Captum Insights可视化多个图像分类推断。
# 
# 首先，让我们收集一些图像，看看模型对它们的看法。
# 为了多样化，我们用猫，茶壶和三叶虫化石的图像：
# 

imgs = ['img/cat.jpg', 'img/teapot.jpg', 'img/trilobite.jpg']

for img in imgs:
    img = Image.open(img)
    transformed_img = transform(img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # 模型需要一个虚拟的批次维度

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')


##########################################################################
# 而且看起来我们的模型可以正确地识别它们--但是，我们想要更深入地挖掘。为此，我们使用
# Captum Insights小组件，我们用一个 ``AttributionVisualizer`` 对象对其进行配置，该对象如下。
#  ``AttributionVisualizer`` 希望有成批的数据，所以我们将引入Captum的 ``Batch`` 辅助类。
#  我们特别关注图片，所以也要导入 ``ImageFeature``.
# 
# 我们用以下参数配置 ``AttributionVisualizer`` ：
# 
# -  要检查的模型阵列（在我们的例子中，只有一个）。
# -  一个评分功能，允许Captum Insights从一个模型中抽出前k个预测值。
# -  一个有序的、人类可读的、我们的模型所训练的类别列表
# -  要寻找的特征列表--在我们的例子中，是一个 ``ImageFeature``
# -  一个数据集，它是一个可迭代的对象，返回成批的输入和标签--就像你在训练中使用的那样。
# 

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# Baseline 是所有全零的输入--这可能会因你的数据而有所不同
def baseline_func(input):
    return input * 0

# 合并我们上面的图像变换
def full_img_transform(input):
    i = Image.open(input)
    i = transform(i)
    i = transform_normalize(i)
    i = i.unsqueeze(0)
    return i


input_imgs = torch.cat(list(map(lambda i: full_img_transform(i), imgs)), 0)

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        )
    ],
    dataset=[Batch(input_imgs, labels=[282,849,69])]
)


#########################################################################
# 请注意，与我们上面的归因不同，我们运行上面的单元格根本没有花费多少时间。这是因为
# Captum Insights让你在一个可视化的小部件中配置不同的归因算法，之后它将计算并显示归因。
# *这个*过程将需要几分钟时间。
# 
# 运行下面的单元格将呈现Captum Insights小组件。然后，你可以选择归因方法及其参数，
# 根据预测类别或预测正确性过滤模型反应，看到模型的预测与相关概率，并查看归因与原始图像的热力图。
# 

visualizer.render()
