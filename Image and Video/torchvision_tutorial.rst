TorchVision物体检测微调教程
====================================================

.. 小贴士::
   为了充分利用这个教程，我们建议使用
   `Colab版本 <https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb>`__.
   这将让你能够对下面介绍的信息进行实验。

在本教程中，我们将在  `宾夕法尼亚州-复旦大学用于行人检测和分割的数据库
 <https://www.cis.upenn.edu/~jshi/ped_html/>`__ 中对一个预先训练好的 `Mask
R-CNN <https://arxiv.org/abs/1703.06870>`__ 模型进行微调。 它包含170张图片，
有345个行人实例，我们将用它来说明如何使用torchvision的新功能，以便在一个自定义的数据
集上训练一个实例分割模型。

定义数据集
--------------------

训练物体检测的参考脚本，实例检测允许fsegmentation和人物关键点，或轻松支持添加新的自定义数据集。
数据集应该继承于标准的 ``torch.utils.data.Dataset`` 类，并实现 ``__len__`` 和
``__getitem__``。

我们唯一的要求是，数据集 ``__getitem__`` 应该返回：

-  图像: 一个大小为 ``(H, W)`` 的PIL图像。
-  目标: 一个包含以下字段的dict

   -  ``boxes (FloatTensor[N, 4])``: 以 ``[x0, y0, x1, y1]`` 格式表示的 ``N`` ，
      个边界框，范围从 ``0`` 到``W`` ， ``0`` 到 ``H``。
   -  ``labels (Int64Tensor[N])``: 每个边界框的标签。``0`` 代表的总是背景类。
   -  ``image_id (Int64Tensor[1])``: 一个图像标识符。它在数据集中的所有图像之间是唯一的，
      并在评估时使用。
   -  ``area (Tensor[N])``: 边界框的面积。这是在用COCO指标评估时使用的，
      以区分小、中、大框之间的指标分数。
   -  ``iscrowd (UInt8Tensor[N])``: iscrowd=True的实例在评估时将被忽略。
   -  (可选) ``masks (UInt8Tensor[N, H, W])``: 每个对象的分割掩码。
   -  (可选) ``keypoints (FloatTensor[N, K, 3])``: N个对象中的每一个
      都包含了K个``[x, y, visibility]`` 格式的关键点，定义了该对象。
      visibility=0意味着该关键点不可见。注意，对于数据增强，翻转关键点的概念取决
      于数据表示法，你应该为你的新关键点表示调整 ``references/detection/transforms.py`` 。

如果你的模型返回上述方法，它们将使它在训练和评估中都能发挥作用，并将使用来自 ``pycocotools``
的评估脚本，这些脚本可以通过 ``pip install pycocotools`` 安装。

.. 小贴士 ::
  对于Windows用户，请从 `gautamchitnis <https://github.com/gautamchitnis/cocoapi>`__
  安装 ``pycocotools`` 命令如下，
  ``pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI``

关于 ``标签`` 的一个说明。该模型认为 ``0`` 类是背景。如果你的数据集不包含背景类，你的 ``标签``
中就不应该有 ``0`` 。例如，假设你只有两个类， *猫* 和 *狗* ，你可以定义``1``（而不是``0``）
来代表 *猫* ， ``2`` 来代表 *狗* 。因此，举例来说，如果其中一个图像有两个类别，你的 ``标签``
张量应该看起来像 ``[1,2]`` 。

此外，如果你想在训练过程中使用长宽比分组（以便每个批次只包含长宽比相似的图像），那么建议同时实现一个
``get_height_and_width`` 方法，它返回图像的高度和宽度。如果没提供这个方法，我们将通过 ``__getitem__``
查询数据集的所有元素，这将在内存中加载图像，比提供自定义方法要慢。

为PennFudan写一个自定义数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

让我们为PennFudan写一个自定义数据集。在 `下载并解压压缩文件
<https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip>`__ 之后，
有以下文件夹结构：

::

   PennFudanPed/
     PedMasks/
       FudanPed00001_mask.png
       FudanPed00002_mask.png
       FudanPed00003_mask.png
       FudanPed00004_mask.png
       ...
     PNGImages/
       FudanPed00001.png
       FudanPed00002.png
       FudanPed00003.png
       FudanPed00004.png

下面是一对图像和分割掩码的例子

.. image:: ../../_static/img/tv_tutorial/tv_image01.png

.. image:: ../../_static/img/tv_tutorial/tv_image02.png

因此，每幅图像都有一个相应的分割掩码，其中每一种颜色都对应一个不同的实例。让我们为这个数据集写一个
``torch.utils.data.Dataset`` 类。

.. code:: python

   import os
   import numpy as np
   import torch
   from PIL import Image


   class PennFudanDataset(torch.utils.data.Dataset):
       def __init__(self, root, transforms):
           self.root = root
           self.transforms = transforms
           # 加载所有的图像文件，将它们排序
           # 确保它们是一致的
           self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
           self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

       def __getitem__(self, idx):
           # 加载图像和掩码
           img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
           mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
           img = Image.open(img_path).convert("RGB")
           # 注意，我们还没有将掩码转换为RGB,
           # 因为每种颜色都对应着不同的实例
           # 0为背景
           mask = Image.open(mask_path)
           # 将PIL图像转换为一个numpy数组
           mask = np.array(mask)
           # 实例被编码为不同的颜色
           obj_ids = np.unique(mask)
           # 第一个id是背景，所以要去掉它
           obj_ids = obj_ids[1:]

           # 将彩色编码的掩码分成一组二进制掩码
           masks = mask == obj_ids[:, None, None]

           # 获取每个掩码的边界框坐标
           num_objs = len(obj_ids)
           boxes = []
           for i in range(num_objs):
               pos = np.nonzero(masks[i])
               xmin = np.min(pos[1])
               xmax = np.max(pos[1])
               ymin = np.min(pos[0])
               ymax = np.max(pos[0])
               boxes.append([xmin, ymin, xmax, ymax])
               
           # 把所有都转换成 torch.Tensor
           boxes = torch.as_tensor(boxes, dtype=torch.float32)
           # 这里只有一个类
           labels = torch.ones((num_objs,), dtype=torch.int64)
           masks = torch.as_tensor(masks, dtype=torch.uint8)

           image_id = torch.tensor([idx])
           area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
           # 假设所有的实例都不是人群
           iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

           target = {}
           target["boxes"] = boxes
           target["labels"] = labels
           target["masks"] = masks
           target["image_id"] = image_id
           target["area"] = area
           target["iscrowd"] = iscrowd

           if self.transforms is not None:
               img, target = self.transforms(img, target)

           return img, target

       def __len__(self):
           return len(self.imgs)

这就是数据集的全部内容。现在让我们定义一个可以对这个数据集进行预测的模型。

定义你的模型
-------------------

在本教程中，我们将使用 `MaskR-CNN <https://arxiv.org/abs/1703.06870>`__ ，
 它是基于 `Faster R-CNN <https://arxiv.org/abs/1506.01497>`__ 之上的。
Faster R-CNN是一个同时预测图像中潜在物体的边界框和类别分数的模型。

.. image:: ../../_static/img/tv_tutorial/tv_image03.png

Mask R-CNN增加了一个额外的分支到Faster R-CNN，它也预测每个实例的分割掩码。

.. image:: ../../_static/img/tv_tutorial/tv_image04.png

这里有人们可能想要修改torchvision modelzoo中的可用模型的两种情况。第一种情况是
我们想从一个预先训练好的模型开始，只对最后一层进行微调。另一种情况是我们想用一个不同
的模型来替换此模型的主干（例如，为了更快的预测）。

让我们看看在下面的章节中我们将如何实现上述两种情况。

1 - 从预训练的模型进行微调
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

假设你想从一个在COCO上预训练的模型开始，并想针对你的特定类别对它进行微调。这里有一个可行的方法：

.. code:: python

   import torchvision
   from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

   # 加载一个在COCO上预训练的模型
   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

   # 用一个新的分类器替换，这个分类器的num_classes是用户定义的。
   num_classes = 2  # 1 class (person) + background
   # 获得分类器的输入特征数量
   in_features = model.roi_heads.box_predictor.cls_score.in_features
   # 用一个新的head来代替预先训练好的head
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

2 - 修改模型以增加不同的主干
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import torchvision
   from torchvision.models.detection import FasterRCNN
   from torchvision.models.detection.rpn import AnchorGenerator

   # 加载一个预先训练好的模型进行分类，只返回特征。
   backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
   # FasterRCNN需要知道主干网中输出通道的数量。
   # 对于mobilenet_v2，它是1280，所以我们需要在这里添加它
   backbone.out_channels = 1280

   # 我们让RPN在每个空间位置生成5×3个锚点，有5种不同的尺寸和3种不同的长宽比。
   # 我们有一个Tuple[Tuple[int]]，因为每个特征图有可能有不同的尺寸和长宽比。
   anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                      aspect_ratios=((0.5, 1.0, 2.0),))

   # 我们来定义什么是我们将用来执行兴趣区域裁剪的特征图，以及重新缩放后的裁剪尺寸。
   #如果你的主干网返回一个张量，featmap_names应该是[0]。
   #一般来说，主干网应该返回一个OrderedDict[Tensor]，而在featmap_names中，你可以选择使用哪些特征图来使用。
   roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                   output_size=7,
                                                   sampling_ratio=2)

   # 在FasterRCNN模型中拼凑碎片
   model = FasterRCNN(backbone,
                      num_classes=2,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)

宾夕法尼亚州数据集的实例分割模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在我们的例子中，我们想从一个预训练的模型中进行微调，鉴于我们的数据集非常小，所以我们将采用1号方法。

在这里，我们也想计算实例分割的掩码，所以我们将使用掩码R-CNN：

.. code:: python

   import torchvision
   from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
   from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


   def get_model_instance_segmentation(num_classes):
       # 加载在COCO上预先训练的实例分割模型
       model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

       # 获得分类器的输入特征数量
       in_features = model.roi_heads.box_predictor.cls_score.in_features
       # 用一个新的head来代替预先训练好的head
       model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

       # 现在得到掩码分类器的输入特征的数量
       in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
       hidden_layer = 256
       # 并用一个新的掩码预测器来取代掩码预测器
       model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                          hidden_layer,
                                                          num_classes)

       return model

这将使 ``模型`` 准备好在你的自定义数据集上进行训练和评估。

把所有放在一起
---------------------------

在``references/detection/``中，我们有一些辅助函数来简化训练和评估检测模型。
在这里，我们将使用 ``references/detection/engine.py`` , ``references/detection/utils.py``
和 ``references/detection/transforms.py`` 。只要把 ``references/detection``
下的所有内容复制到你的文件夹，并在这里使用它们。

让我们写一些用于数据增强/转换的辅助函数:

.. code:: python

   import transforms as T

   def get_transform(train):
       transforms = []
       transforms.append(T.PILToTensor())
       transforms.append(T.ConvertImageDtype(torch.float))
       if train:
           transforms.append(T.RandomHorizontalFlip(0.5))
       return T.Compose(transforms)


测试 ``forward()``方法(可选)
---------------------------------------

在对数据集进行迭代之前，最好观察模型在样本数据的训练和推理时间内的期望值。

.. code:: python

   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
   dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
   data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
   # 用于训练
   images,targets = next(iter(data_loader))
   images = list(image for image in images)
   targets = [{k: v for k, v in t.items()} for t in targets]
   output = model(images,targets)   # 返回损失和检测
   # 用于推理
   model.eval()
   x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
   predictions = model(x)           # 返回预测

现在让我们来写一个主函数，来执行训练和验证：

.. code:: python

   from engine import train_one_epoch, evaluate
   import utils


   def main():
       # 在GPU上训练，如果没有GPU，则在CPU上训练
       device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

       # 我们的数据集只有两种类--背景和人物
       num_classes = 2
       # 使用我们的数据集和定义的转换
       dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
       dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

       # 将数据集分成训练集和测试集
       indices = torch.randperm(len(dataset)).tolist()
       dataset = torch.utils.data.Subset(dataset, indices[:-50])
       dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

       # 定义训练和验证数据加载器
       data_loader = torch.utils.data.DataLoader(
           dataset, batch_size=2, shuffle=True, num_workers=4,
           collate_fn=utils.collate_fn)

       data_loader_test = torch.utils.data.DataLoader(
           dataset_test, batch_size=1, shuffle=False, num_workers=4,
           collate_fn=utils.collate_fn)

       # 使用我们的辅助函数获得模型
       model = get_model_instance_segmentation(num_classes)

       # 将模型移到正确的设备上
       model.to(device)

       # 构建一个优化器
       params = [p for p in model.parameters() if p.requires_grad]
       optimizer = torch.optim.SGD(params, lr=0.005,
                                   momentum=0.9, weight_decay=0.0005)
       # and a learning rate scheduler
       lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=3,
                                                      gamma=0.1)

       # 让我们对它进行10个epoch的训练
       num_epochs = 10

       for epoch in range(num_epochs):
           # 训练一个epoch，每10次迭代打印一次
           train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
           # 更新学习率
           lr_scheduler.step()
           # 在测试数据集上进行评估
           evaluate(model, data_loader_test, device=device)

       print("That's it!")

You should get as output for the first epoch:

::

   Epoch: [0]  [ 0/60]  eta: 0:01:18  lr: 0.000090  loss: 2.5213 (2.5213)  loss_classifier: 0.8025 (0.8025)  loss_box_reg: 0.2634 (0.2634)  loss_mask: 1.4265 (1.4265)  loss_objectness: 0.0190 (0.0190)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 1.3121  data: 0.3024  max mem: 3485
   Epoch: [0]  [10/60]  eta: 0:00:20  lr: 0.000936  loss: 1.3007 (1.5313)  loss_classifier: 0.3979 (0.4719)  loss_box_reg: 0.2454 (0.2272)  loss_mask: 0.6089 (0.7953)  loss_objectness: 0.0197 (0.0228)  loss_rpn_box_reg: 0.0121 (0.0141)  time: 0.4198  data: 0.0298  max mem: 5081
   Epoch: [0]  [20/60]  eta: 0:00:15  lr: 0.001783  loss: 0.7567 (1.1056)  loss_classifier: 0.2221 (0.3319)  loss_box_reg: 0.2002 (0.2106)  loss_mask: 0.2904 (0.5332)  loss_objectness: 0.0146 (0.0176)  loss_rpn_box_reg: 0.0094 (0.0123)  time: 0.3293  data: 0.0035  max mem: 5081
   Epoch: [0]  [30/60]  eta: 0:00:11  lr: 0.002629  loss: 0.4705 (0.8935)  loss_classifier: 0.0991 (0.2517)  loss_box_reg: 0.1578 (0.1957)  loss_mask: 0.1970 (0.4204)  loss_objectness: 0.0061 (0.0140)  loss_rpn_box_reg: 0.0075 (0.0118)  time: 0.3403  data: 0.0044  max mem: 5081
   Epoch: [0]  [40/60]  eta: 0:00:07  lr: 0.003476  loss: 0.3901 (0.7568)  loss_classifier: 0.0648 (0.2022)  loss_box_reg: 0.1207 (0.1736)  loss_mask: 0.1705 (0.3585)  loss_objectness: 0.0018 (0.0113)  loss_rpn_box_reg: 0.0075 (0.0112)  time: 0.3407  data: 0.0044  max mem: 5081
   Epoch: [0]  [50/60]  eta: 0:00:03  lr: 0.004323  loss: 0.3237 (0.6703)  loss_classifier: 0.0474 (0.1731)  loss_box_reg: 0.1109 (0.1561)  loss_mask: 0.1658 (0.3201)  loss_objectness: 0.0015 (0.0093)  loss_rpn_box_reg: 0.0093 (0.0116)  time: 0.3379  data: 0.0043  max mem: 5081
   Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2540 (0.6082)  loss_classifier: 0.0309 (0.1526)  loss_box_reg: 0.0463 (0.1405)  loss_mask: 0.1568 (0.2945)  loss_objectness: 0.0012 (0.0083)  loss_rpn_box_reg: 0.0093 (0.0123)  time: 0.3489  data: 0.0042  max mem: 5081
   Epoch: [0] Total time: 0:00:21 (0.3570 s / it)
   creating index...
   index created!
   Test:  [ 0/50]  eta: 0:00:19  model_time: 0.2152 (0.2152)  evaluator_time: 0.0133 (0.0133)  time: 0.4000  data: 0.1701  max mem: 5081
   Test:  [49/50]  eta: 0:00:00  model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)  time: 0.0735  data: 0.0022  max mem: 5081
   Test: Total time: 0:00:04 (0.0828 s / it)
   Averaged stats: model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)
   Accumulating evaluation results...
   DONE (t=0.01s).
   Accumulating evaluation results...
   DONE (t=0.01s).
   IoU metric: bbox
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.780
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.672
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.672
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.755
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
   IoU metric: segm
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.704
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.748
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.749
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758

因此，经过一个epoch的训练，我们得到了一个COCO式的mAP为60.6，而掩码mAP为70.4。

在训练了10个epochs之后，我得到了以下指标：

::

   IoU metric: bbox
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.935
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.844
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.844
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.777
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.870
   IoU metric: segm
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.919
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.799
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818

但预测是什么样子的呢？让我们从数据集中的一张图片出发，并且验证。

.. image:: ../../_static/img/tv_tutorial/tv_image05.png

The trained model predicts 9
instances of person in this image, let’s see a couple of them: 

.. image:: ../../_static/img/tv_tutorial/tv_image06.png

.. image:: ../../_static/img/tv_tutorial/tv_image07.png

结果看起来相当不错!

收尾工作
-----------

在本教程中，你已经学会了如何在自定义数据集上为实例分割模型创建自己的训练管道。为此，
你写了一个 ``torch.utils.data.Dataset`` 类，该类返回图像、地表真值框和分割掩码。
你还利用了在COCO train2017上预训练的Mask R-CNN模型，以便在这个新数据集上进行转移学习。

对于一个更完整的例子，包括多机器/多gpu训练，请查看 ``references/detection/train.py``，
它存在与torchvision repo中。

你可以在 `这里 <https://pytorch.org/tutorials/_static/tv-training-code.py>`__ 下载本教程的完整源文件。

   

