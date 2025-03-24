import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
from cv2 import imread, imwrite

from pydensecrf.utils import unary_from_labels

def CRFs(original_image, predicted_image, sxy1=2, sxy2=8, srgb=8):
    img = original_image
    anno_rgb = predicted_image.astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))

    # 使用densecrf2d类
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # 得到一元势（负对数概率）
    U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
    # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
    d.setUnaryEnergy(U)

    # 增加了与颜色无关的术语，只是位置-----会惩罚空间上孤立的小块分割,即强制执行空间上更一致的分割
    d.addPairwiseGaussian(sxy=sxy1, compat=8)

    # 增加了颜色相关术语，即特征是(x,y,r,g,b)-----使用局部颜色特征来细化它们
    d.addPairwiseBilateral(sxy=sxy2, srgb=srgb, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)
    '''
    addPairwiseGaussian函数里的sxy为公式中的 $\theta_{\gamma}$,  
    addPairwiseBilateral函数里的sxy、srgb为$\theta_{\alpha}$ 和 $\theta_{\beta}$
    '''

    # 进行10次推理
    Q = d.inference(10)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP, :]
    crf_img = MAP.reshape(img.shape)
    crf_img = cv2.bitwise_not(crf_img)
    return crf_img