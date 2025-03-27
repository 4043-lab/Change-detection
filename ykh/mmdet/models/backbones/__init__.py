# Copyright (c) OpenMMLab. All rights reserved.
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .swin import SwinTransformer
from .resnet_cd import ResNet_Cd
from .resnet_cd2 import ResNet_Cd2
# from .clip_models import CLIPResNetWithAttention,CLIPTextContextEncoder,ContextDecoder
from .clip_backbone2 import CLIPResNet, CLIPResNetWithAttention, CLIPVisionTransformer, CLIPTextEncoder, CLIPTextContextEncoder, ContextDecoder


__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'ResNeSt', 'SwinTransformer',
    'PyramidVisionTransformer', 'PyramidVisionTransformerV2',
    'ResNet_Cd','ResNet_Cd2',
    'CLIPResNetWithAttention','CLIPTextContextEncoder','ContextDecoder'
]
