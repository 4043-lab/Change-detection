# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_free_head import AnchorFreeHead
from .anchor_head import AnchorHead
from .condinst_head import CondInstBoxHead, CondInstSegmHead, CondInstMaskBranch, CondInstMaskHead
from .box_solov2_head import BoxSOLOv2Head
from .discobox_head import DiscoBoxSOLOv2Head, DiscoBoxMaskFeatHead
from .box2mask_head import Box2MaskHead
from .box_solov2_head_v2 import BoxSOLOv2Head_v2
from .box_solov2_head_v3 import BoxSOLOv2Head_v3
from .box_solov2_head_v4 import BoxSOLOv2Head_v4
from .box2cd_clip_head import Box2cd_CLIP_Head
from .box2cd_clip_head_v2 import Box2cd_CLIP_Head_v2



__all__ = [
    'AnchorFreeHead', 'AnchorHead', 'CondInstBoxHead', 'CondInstMaskBranch',
    'CondInstMaskHead', 'CondInstSegmHead', 'BoxSOLOv2Head', 'DiscoBoxSOLOv2Head',
    'DiscoBoxMaskFeatHead', 'Box2MaskHead',
    'BoxSOLOv2Head_v2','BoxSOLOv2Head_v3','BoxSOLOv2Head_v4',
    'Box2cd_CLIP_Head','Box2cd_CLIP_Head_v2',

]
