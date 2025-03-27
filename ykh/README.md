# 代码介绍
## 1、代码环境和依赖

``` markdown
Linux or macOS
Python 3.6+ (Python 3.8 in our envs)
PyTorch 1.7+ (1.9.0 in our envs)
CUDA 9.2+ (CUDA 11.1 in our envs)
GCC 5+
mmdet==2.25.0
mmcv-full==1.5.0
mmseg==0.20.0
regex
ftfy
fvcore
timm==0.6.13 
```
## 2、安装指令
本项目基于boxinstseg完成
安装环境可以参考 https://github.com/LiWentomng/BoxInstSeg
``` python
conda create -n box2cd_clip python=3.8 -y
conda activate box2cd_clip

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -U openmim
mim install mmcv-full==1.5.0

git clone https://github.com/LiWentomng/BoxInstSeg.git
cd BoxInstSeg

bash setup.sh   #compile the whole envs

pip install mmsegmentation==0.20.0

pip install regex
pip install ftfy
pip install fvcore
pip install timm==0.6.13
pip install scikit-image
```

## 3、代码说明
* 代码配置文件目录见：`BoxInstSeg/configs/changeDetection/**`带目录下包含`box2cd`和`box2cd_clip` 其对应为基于水平集演化的弱监督模型Box2Change以及利用CLIP实现跨模态的弱监督模型。
* 模型框架、Head已经loss见：`BoxInstSeg/mmdet/models/**`
* 数据集格式见：`BoxInstSeg/mmdet/datasets/levir_cd.py`

## 4、 数据集样式
本代码数据集目录格式如下：
```
├─annotations
├─train
│  ├─A
│  ├─B
│  └─label
└─val
    ├─A
    ├─B
    └─label
```
其中`annotaions`内包含box级标签用于训练，`label`下为像素级标签，用于实验比较结果。
`annotaions `下位json文件，其内格式符合coco数据集样式。
## 5、代码指令
训练指令
```
格式： python tools/train.py 配置文件.py
示例：
python  tools/train.py  configs/my_config/box2cd_clip_ablation/box2cd_clip_resnet_whu.py
```
样本生成
```
格式： python box2cd_result.py  配置文件.py  模型文件路径  --show-dir 图片输出目录
示例：
python box2cd_result.py configs/my_config/box2cd_self/box2cd_self_GZ_base.py  work_dirs/box2cd_self_LEVIR_sscl/epoch_24.pth  --show-dir  work_dirs/result_cross_dataset/result_self_gz_sscl/
```

