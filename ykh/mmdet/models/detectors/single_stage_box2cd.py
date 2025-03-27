'''
自定义的 以box为弱监督的框架
'''
import torch.nn as nn
import torch
import warnings
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import bbox2result
import numpy as np


@DETECTORS.register_module()
class SingleStageBox2CdDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageBox2CdDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img1,img2):
        x = self.backbone(img1,img2)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img1,img2):
        x = self.extract_feat(img1,img2)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img1,
                      img2,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img1,img2)
        outs = self.bbox_head(x)

        # outs_e=self.bbox_head(x,eval=True)
        # print('*'*30)
        # for i in outs_e:
        #     print("%"*20)
        #     for j in i:
        #         print(' ',j.size())
        # seg_info=outs_e+(img_metas, self.test_cfg)
        # seg_result=self.bbox_head.get_seg(*seg_info)


        ###loss是否也需要2个图片？?
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img1-img2, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_test(self, img_1,img_2, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(img_1, 'img_1'), (img_2, 'img_2'),(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img_1)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img_1)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(img_1, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(img_1[0],img_2[0], img_metas[0], **kwargs)
        else:
            assert img_1[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{img_1[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test( img_1,img_2, img_metas, **kwargs)

    def forward(self, img_1,img_2, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img_1[0], img_metas[0])

        if return_loss:
            return self.forward_train(img_1,img_2, img_metas, **kwargs)
        else:
            return self.forward_test(img_1,img_2, img_metas, **kwargs)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        # print('tran step data:',data)
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def simple_test(self, img1,img2, img_meta, rescale=False):
        x = self.extract_feat(img1,img2)
        outs = self.bbox_head(x, eval=True)

        seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        results_list = self.bbox_head.get_seg(*seg_inputs)
        format_results_list = []
        for results in results_list:
            format_results_list.append(self.format_results(results))
        return format_results_list

    def format_results(self, results):
        bbox_results = [[] for _ in range(self.bbox_head.num_classes)]
        mask_results = [[] for _ in range(self.bbox_head.num_classes)]
        score_results = [[] for _ in range(self.bbox_head.num_classes)]

        for cate_label, cate_score, seg_mask in zip(results.labels, results.scores, results.masks):
            if seg_mask.sum() > 0:
                mask_results[cate_label].append(seg_mask.cpu())
                score_results[cate_label].append(cate_score.cpu())
                ys, xs = torch.where(seg_mask)
                min_x, min_y, max_x, max_y = xs.min().cpu().data.numpy(), ys.min().cpu().data.numpy(), xs.max().cpu().data.numpy(), ys.max().cpu().data.numpy()
                bbox_results[cate_label].append([min_x, min_y, max_x + 1, max_y + 1, cate_score.cpu().data.numpy()])

        bbox_results = [np.array(bbox_result) if len(bbox_result) > 0 else np.zeros((0, 5)) for bbox_result in
                        bbox_results]

        return bbox_results, (mask_results, score_results)

    def aug_test(self, img_1,img_2, img_metas, rescale=False):
        raise NotImplementedError


