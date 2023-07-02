"""
优化用于部署的视觉转换器模型
===========================

`Jeff Tang <https://github.com/jeffxtang>`_,
`Geeta Chauhan <https://github.com/gchauhan/>`_

视觉转化器模型应用了自然语言处理中引入的基于注意力的尖端的的转化器模型，该模型在自然语言处理中被引入，以实现各种最先进的（SOTA）结果。
所有种类的最先进（SOTA）的结果，到计算机视觉任务。 Facebook数据高效的图像变换器 `DeiT <https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification>`_
是一个在ImageNet上训练的视觉转化器模型，用于图像分类。

在本教程中，我们将首先介绍什么是DeiT以及如何使用它、然后通过脚本、量化、优化的完整步骤、
并在iOS和Android应用程序中使用该模型。我们还将比较量化、优化和非量化、非优化模型的性能。
并展示将量化和优化应用于模型的好处。


"""



###############################################################################
# 什么是 DeiT
# ---------------------
#
# 卷积神经网络（CNN）自2012年深度学习兴起以来，一直是图像分类的主要模型。但CNN通常
#  需要数以亿计的图像进行训练才能达到SOTA结果。
#  DeiT是一个视觉转化器模型，它需要更少的数据和计算资源进行训练，以便与领先的CNNs进行图像分类，这是通过以下两个方面实现的：
#
# -  数据增强，模拟在更大的数据集上进行训练；
# -  蒸馏法允许变换器网络从CNN的结果中学习；
#
# DeiT表明，在数据和资源有限的情况下，Transformers可以成功应用于计算机视觉任务。
# 关于更多 更多关于DeiT的细节，请看`repo <https://github.com/facebookresearch/deit>`_
# 和 `paper <https://arxiv.org/abs/2012.12877>`_.
#


######################################################################
# 用DeiT对图像进行分类
# -------------------------------
#
# 追随 ``README.md`` 关于如何使用DeiT对图像进行分类的详细信息，请访问DeiT资源库，或者进行快速测试，首先安装
# 所需的软件包：
#
# .. code-block:: python
#
#    pip install torch torchvision timm pandas requests

#######################################################
# 要在Google Colab中运行，通过运行以下命令安装依赖：
#
# .. code-block:: python
#
#    !pip install timm pandas requests

#############################
# 然后运行下面的脚本:

from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(torch.__version__)
# 应该是 1.8.0


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
img = transform(img)[None,]
out = model(img)
clsidx = torch.argmax(out)
print(clsidx.item())


######################################################################
# 输出应该是 269,根据类的 ImageNet 列表索引到标签文件 `标签文件 <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`_, 映射到  ``timber
# wolf, grey wolf, gray wolf, Canis lupus``.
#
# 现在我们已经验证了我们可以使用 DeiT 模型进行分类图像，让我们看看如何修改模型，使其可以在 iOS上运行以及使用在安卓软件上。
#


######################################################################
# 为DeiT编写脚本
# ----------------------
# 为了在移动端使用模型，我们首先需要编写脚本。请参阅 `脚本和优化配方法 <https://pytorch.org/tutorials/recipes/script_optimized.html>`_快速概览。
# 运行下面的代码以=来转换可以在移动设备上运行的TorchScript格式。
#


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("fbdeit_scripted.pt")


######################################################################
#  大小约为 346MB 的脚本模型文件 ``fbdeit_scripted.pt`` 被生成。
#


######################################################################
# 量化 DeiT
# ---------------------
# 显著减小训练模型大小，同时保持推理精度大致相同，量化可以应用于模型。
# 由于DeiT中使用的transformer模型，我们可以轻松地将动态量化应用于模型，因为动态量化最适合 LSTM 和transformer模型（有关更多详细信息，
# 请参阅`此处 <https://pytorch.org/docs/stable/quantization.html?highlight=quantization#dynamic-quantization>`_）
#
# 现在运行下面的代码：
#

# 使用 'x86' 进行服务推理（之前的 'fbgemm' 仍然是可用的，但是 'x86' 是默认的推荐），``qnnpack`` 用于移动端推理。
backend = "x86" # 用``qnnpack``代替，在这个笔记本上，量化模型的推理速度要差很多
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("fbdeit_scripted_quantized.pt")


######################################################################
# 这将生成模型的脚本化和量化版本
# ``fbdeit_quantized_scripted.pt``, 大小约为89MB，比非量化的模型大小减少74%! 非量化的模型大小为346MB!
#

######################################################################
# 你可以使用``scripted_quantized_model``来生成相同的
# 推理结果：
#

out = scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# 应打印相同的输出269

######################################################################
# 优化DeiT
# ---------------------
# 在使用量化和脚本化的模型前的最后一步是对其进行优化：
#

from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
optimized_scripted_quantized_model.save("fbdeit_optimized_scripted_quantized.pt")


######################################################################
# 生成的 "fbdeit_optimized_scripted_quantized.pt "文件的大小与量化后的非优化模型的大小差不多。 推理结果仍然是一样的。
#



out = optimized_scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# 同样，应该打印相同的输出269


######################################################################
# 使用精简版解释器
# ------------------------
#
# 为了看一看模型大小的减少和推理速度的提高，Lite 解释器的结果，让我们创建一个精简版的模型。
#

optimized_scripted_quantized_model._save_for_lite_interpreter("fbdeit_optimized_scripted_quantized_lite.ptl")
ptl = torch.jit.load("fbdeit_optimized_scripted_quantized_lite.ptl")


######################################################################
# 虽然精简版的模型大小与非精简版相当，但当 在手机上运行精简版时，推理速度会有所提高。
#


######################################################################
# 比较推理速度
# ---------------------------
#
# 要看四个模型--原始模型、脚本模型、量化和脚本模型--的推理速度有什么不同，运行下面的代码：
#

with torch.autograd.profiler.profile(use_cuda=False) as prof1:
    out = model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof2:
    out = scripted_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof3:
    out = scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof4:
    out = optimized_scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof5:
    out = ptl(img)

print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
print("scripted model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))
print("scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))
print("scripted & quantized & optimized model: {:.2f}ms".format(prof4.self_cpu_time_total/1000))
print("lite model: {:.2f}ms".format(prof5.self_cpu_time_total/1000))

######################################################################
# 在谷歌Colab上运行的结果是：
#
# ::
#
#    original model: 1236.69ms
#    scripted model: 1226.72ms
#    scripted & quantized model: 593.19ms
#    scripted & quantized & optimized model: 598.01ms
#    lite model: 600.72ms
#


######################################################################
# 下面的结果总结了每个模型所花费的推理时间
# 以及每个模型相对于原始模型的减少的百分比。

#

import pandas as pd
import numpy as np

df = pd.DataFrame({'Model': ['original model','scripted model', 'scripted & quantized model', 'scripted & quantized & optimized model', 'lite model']})
df = pd.concat([df, pd.DataFrame([
    ["{:.2f}ms".format(prof1.self_cpu_time_total/1000), "0%"],
    ["{:.2f}ms".format(prof2.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof2.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    ["{:.2f}ms".format(prof3.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof3.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    ["{:.2f}ms".format(prof4.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof4.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    ["{:.2f}ms".format(prof5.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof5.self_cpu_time_total)/prof1.self_cpu_time_total*100)]],
    columns=['Inference Time', 'Reduction'])], axis=1)

print(df)

"""
        Model                             Inference Time    Reduction
0	original model                             1236.69ms           0%
1	scripted model                             1226.72ms        0.81%
2	scripted & quantized model                  593.19ms       52.03%
3	scripted & quantized & optimized model      598.01ms       51.64%
4	lite model                                  600.72ms       51.43%
"""

######################################################################
# 学习更多
# ~~~~~~~~~~~~~~~~~
#
# - `Facebook Data-efficient Image Transformers <https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification>`__
# - `Vision Transformer with ImageNet and MNIST on iOS <https://github.com/pytorch/ios-demo-app/tree/master/ViT4MNIST>`__
# - `Vision Transformer with ImageNet and MNIST on Android <https://github.com/pytorch/android-demo-app/tree/master/ViT4MNIST>`__
