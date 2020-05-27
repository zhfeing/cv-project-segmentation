# 图像处理与建模说明文档

## 实现内容

复现图像语义分割模型DeepLabV3并在Pascal Voc和Cityscapes数据集进行测试。

## 运行环境

| | |
| ---- | ---- |
| 操作系统 | Ubuntu 18.04.4 LTS |
| CPU | 双路 Intel(R) Xeon(R) Gold &  5122 CPU @ 3.60GHz|
| GPU | NVIDIA(R) Quadro(R) P6000 |
| 代码语言 | Python 3.7.4 |
| 深度学习框架 | PyTorch-1.5 |

## 文件组织
```
|
├── README.md
├── checkpoints             训练好的模型
│   ├── deeplabv3_resnet101_citys_best_model.pth
│   ├── deeplabv3_resnet101_pascal_voc_best_model.pth
│   ├── deeplabv3_resnet50_citys_best_model.pth
│   └── deeplabv3_resnet50_pascal_voc_best_model.pth
├── data                    数据集模块
│   ├── __init__.py
│   ├── dataloader          加载数据集模块
│   │   ├── __init__.py
│   │   ├── cityscapes.py
│   │   ├── pascal_voc.py
│   │   ├── segbase.py
│   │   └── utils.py
│   └── downloader          下载数据集模块
│       ├── __init__.py
│       ├── cityscapes.py
│       └── pascal_voc.py
├── models                  模型结构
│   ├── __init__.py
│   ├── base_models
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   └── resnetv1b.py
│   ├── deeplabv3.py
│   ├── fcn.py
│   ├── model_store.py
│   └── segbase.py
├── nn                      辅助模型结构
│   ├── __init__.py
│   └── jpu.py
├── runs                    在测试集上运行结果
│   └── pred_pic
│       ├── deeplabv3_resnet101_citys
│       │   ├── frankfurt_000000_000294_leftImg8bit.png
│       │   ├── frankfurt_000000_000576_leftImg8bit.png
|       |   └── ...
│       ├── deeplabv3_resnet101_pascal_voc
│       │   ├── 2007_000033.png
│       │   ├── 2007_000042.png
|       |   └── ...
│       ├── deeplabv3_resnet50_citys
│       │   ├── frankfurt_000000_000294_leftImg8bit.png
│       │   ├── frankfurt_000000_000576_leftImg8bit.png
│       │   └── ...
│       └── deeplabv3_resnet50_pascal_voc
│           ├── 2007_000033.png
│           ├── 2007_000042.png
│           └── ...
├── scripts                 运行demo、验证和训练脚本
│   ├── demo.py
│   ├── eval.py
│   └── train.py
├── train_deeplabv3.sh      训练模型样例脚本
└── utils                   工具
    ├── __init__.py
    ├── distributed.py
    ├── download.py
    ├── filesystem.py
    ├── logger.py
    ├── loss.py
    ├── lr_scheduler.py
    ├── score.py
    └── visualize.py
```