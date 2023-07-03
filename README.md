# swin-transformer-ocr
ocr with [swin-transformer](https://arxiv.org/abs/2103.14030)
## 写在前面
本项目限于作者水平有限，如有纰漏，请不吝指正，我会及时更正。
## 概述
本项目是基于Swin-Transformer骨干网络实现的一款OCR算法，是个人参考[YongWookHa](https://github.com/YongWookHa/swin-transformer-ocr)
的实现，并根据个人理解做出改进后实现的。

本项目得益于[timm](https://github.com/rwightman/pytorch-image-models) and [x_transformers](https://github.com/lucidrains/x-transformers)等开源框架，
可以使我的工作聚焦于OCR业务本身。

本项目的Demo是英文（含标点和空格）OCR识别，可以根据自己的情况迁移到汉字或者手写体。
Demo数据是得益于以往的项目经验纯人工生成，尺寸为32*320，在测试时需要将数据高度等比缩放，宽度根据实际情况添加padding或者裁剪。
而在实际应用中可以根据自己业务需求训练更大长度的模型。


## 性能
在本项目数据集上，测试集准确率约0.96，且未见饱和情况。

## 改进点
1、path size 和 window size 都改为2，维度也对应减少，加快训练速度，同时针对英文小区块（例如ll紧密相连的样本）处理得更好。

2、scheduler，原作的CustomCosineAnnealingWarmupRestarts在我的样本上训练若干epoch后，准确率在0.65左右达到
性能饱和，需要比较好的调参技巧才能继续优化。修改成前期采用CustomCosineAnnealingWarmupRestarts，
性能饱和后采用CosineAnnealingLR，具体调参是需要根据tensorboard信息结合经验来分析。

3、dataset由原先的单文件加载改为lmdb加载，方便数据迁移和复现，本项目里面的lmdb仅为demo。

## 数据准备
```bash
./dataset/
├─ images/
│  ├─ train
│   ├─ image_0.jpg
│   ├─ image_1.jpg
│   ├─ ...
│   ├─ labels.txt
│  ├─ val
│   ├─ image_0.jpg
│   ├─ image_1.jpg
│   ├─ ...
│   ├─ labels.txt
├─ lmdb/
│  ├─ train
│   ├─ data.mdb
│   ├─ lock.mdb
│  ├─ val
│   ├─ data.mdb
│   ├─ lock.mdb   

create_lmdb_dataset.py生成lmdb相关文件
# in labels.txt
cropped_image_0.jpg Hello World.
cropped_image_1.jpg vision-transformer-ocr
...
```

需要先根据自己的需求生成训练文件，然后执行create_lmdb_dataset.py生成lmdb相关文件。

## 配置项
在 `settings/` 目录下, 找到 `demo.yaml`。 可以根据自己的需要设置超参数，前提是足够了解哈。

## 训练
```bash
python train.py --version 0 --setting settings/demo.yaml --num_workers 16 --batch_size 64
```
我的数据量大概有150万个样本，在3090上训练加调参共计用了2天左右完成，可以作为大家的参考。
关注tensorboard日志信息，调参很重要。  
```
tensorboard --log_dir tb_logs --bind_all
```  

## 预测  
当完成训练后，可以通过预测代码评估自己的模型在真实数据集上的表现。需要注意的是，swin-transformer是序列定长的，
所以在处理实际问题时需要根据需要padding到固定长度（比如本项目中的32*320）。也可以训练一个大长度的模型，只是需要
更长的时间。

```  
python predict.py --setting <your_setting.yaml> 
```

## 导出 ONNX  
导出Onnx非常简单，非本项目重点，有兴趣的请参考[related pytorch-lightning document](https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html).

## 最后
鼓励开源精神，尊重知识产权，之前因为工作关系，许多东西都无法分享，希望在将来能做更多的东西出来。
```
```
