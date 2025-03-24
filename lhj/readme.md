# 代码介绍
## 1.dataset文件夹
存放数据集。数据集内部通常按time1、time2、label的格式存放即可。

## 2.heatmao文件夹
存放网络中间特征可视化热力图。运行heatmap.py文件可以得到结果。

## 3.model文件夹
存放网络模型代码。

## 4.result文件夹
存放网络训练后得到的参数文件。

## 5.test_image_all文件夹
存放验证集结果。内部第一级为数据集，第二级为模型，第三级为可视化文件，包括黑白的原始cd结果和彩色的带tp/tn/fp/fm结果

## 6.utils文件夹
包含网络训练过程中的工具代码

## 7.generate_edges_canny.py
利用原始cd标签生成边缘标签

## 8.heatmap.py
用于生成网络中间特征可视化热力图

## 9.test_vis_color_all.py
全监督模型验证、可视化

## 10.test_vis_color_all_CAM.py
基于CAM的弱监督模型验证、可视化

## 11.train_cls.py
基于图像级标签的弱监督模型训练

## 12.train.py
全监督模型训练

# 代码运行
    # 全监督模型
    python train.py
    # 弱监督模型
    python train_cls.py
- 并未设置命令行参数，模型相关参数请自行修改代码，包含GPU使用、模型超参数等