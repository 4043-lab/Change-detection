'''
自定义的 以box为弱监督的框架
'''
from .single_stage_box2cd import SingleStageBox2CdDetector
from ..builder import DETECTORS

from mmdet.core.visualization import imshow_det_cdmask
import mmcv
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms


@DETECTORS.register_module()
class Box2CdLevelSet(SingleStageBox2CdDetector):
    r"""Implementation of `Box-supervised Instance Segmentation
    with Level Set Evolution <https://arxiv.org/abs/2207.09055.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Box2CdLevelSet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def show_cd_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # not empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_cdmask(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def affine_transform(self, img_torch):
        # 最后一维度翻转
        trans = transforms.RandomHorizontalFlip(p=1)
        dim = -1
        if isinstance(img_torch, tuple):
            output = list(img_torch)
            for i in range(len(img_torch)):
                output[i] = trans(img_torch[i]).contiguous()
        else:
            dim = -1
            indices = [slice(None)] * img_torch.dim()
            indices[dim] = torch.arange(img_torch.size(dim) - 1, -1, -1,
                                        dtype=torch.long, device=img_torch.device)
            output = img_torch[tuple(indices)]

        # output = F.interpolate(img_torch, scale_factor=0.5, mode='bilinear')
        return output

    def forward_train(self,
                      img1,
                      img2,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img1, img2)

        _x = self.affine_transform(x)

        outs = self.bbox_head(x)
        ins_pred, cate_pred, levelset_feats = outs
        ins_pred2, cate_pred2, levelset_feats2 = self.bbox_head(_x)

        fpn_feature = (ins_pred, ins_pred2)

        # loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks,img1-img2, img_metas,self.train_cfg)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks,img1-img2, img_metas,self.train_cfg,fpn_feature)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

