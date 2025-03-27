'''
自定义的 以box为弱监督的利用clip进行跨模态训练框架   双分支  changeCLip的融合方式 非 backbone阶段融合
'''
from .single_stage_box2cd import SingleStageBox2CdDetector
from ..builder import DETECTORS

from mmdet.core.visualization import imshow_det_cdmask
import mmcv
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
# from mmseg.models import builder
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .untils import tokenize
from ._blocks import ConvTransposed3x3,CrossAttention

import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib



class BCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(BCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, mask, target):
        if mask.dim() > 2:
            mask = mask.view(mask.size(0), mask.size(1), -1)  # N,C,H,W => N,C,H*W
            mask = mask.transpose(1, 2)  # N,C,H*W => N,H*W,C
            mask = mask.contiguous().view(-1, mask.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        logpt = F.log_softmax(mask, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


@DETECTORS.register_module()
class Box2Cd_CLIP_vit(SingleStageBox2CdDetector):
    r"""Implementation of `Box-supervised Instance Segmentation
    with Level Set Evolution <https://arxiv.org/abs/2207.09055.pdf>`_."""

    def __init__(self,
                 backbone,
                 bbox_head,
                 neck,
                 text_encoder,
                 context_decoder,
                 class_names,
                 context_length,
                 score_thr=0.5,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 token_embed_dim=512, text_dim=512,
                 **args
                 ):
        super(Box2Cd_CLIP_vit,self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, None, init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'

            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.text_encoder = build_backbone(text_encoder)
        self.context_decoder = build_backbone(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
        self.score_thr = score_thr

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = build_neck(neck)

        # self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        # assert self.with_decode_head
        self.cross_layer = []
        self.conv_1_1_layer = []
        self.conv_1_2_layer = []
        self.up_layer=[]
        # channels=[256,512,1024,2048]
        channels=neck.in_channels
        text_len=channels[2]-768
        wh_list=[64,32,16,8]
        for i, channel in enumerate(channels):
            upsample=ConvTransposed3x3(text_len,text_len,output_padding=1)
            upsample_name=f'upsample_{i + 1}'
            self.add_module(upsample_name,upsample)
            self.up_layer.append(upsample_name)

            # Cross = CrossAttention(dropout=0.5,h=wh_list[i],w=wh_list[i],feature_dim=256)
            Cross = CrossAttention(dropout=0.5, h=wh_list[i], w=wh_list[i], feature_dim=channel,d_model=512)
            Cross_name = f'Cross_{i + 1}'
            self.add_module(Cross_name, Cross)
            self.cross_layer.append(Cross_name)


            # 相减或融合模块
            conv_1_1 = nn.Sequential(
                nn.Conv2d(in_channels=channel*2, out_channels=channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(channel, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            )
            conv_1_1_name = f'conv_1_1_{i + 1}'
            self.add_module(conv_1_1_name, conv_1_1)
            self.conv_1_1_layer.append(conv_1_1_name)

            conv_1_2 = nn.Sequential(
                nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(channel, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            )
            conv_1_2_name = f'conv_1_2_{i + 1}'
            self.add_module(conv_1_2_name, conv_1_2)
            self.conv_1_2_layer.append(conv_1_2_name)


        self.neg_conv_layer=nn.Sequential(
            ConvTransposed3x3(256, 256, output_padding=1),
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            ConvTransposed3x3(128, 128, output_padding=1),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            ConvTransposed3x3(64, 64, output_padding=1),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
                nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            ConvTransposed3x3(32, 32, output_padding=1),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1),
                nn.BatchNorm2d(16, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            ConvTransposed3x3(16, 16, output_padding=1),
                nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1),
                nn.BatchNorm2d(2, eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU(),
            )


    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                self.auxiliary_head = build_head(auxiliary_head)

    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = build_head(identity_head)

    def extract_feat(self, img1, img2):
        """Extract features from images. 直接相减融合"""
        x1 = self.backbone(img1)
        x2 = self.backbone(img2)

        return x1,x2



    def affine_transform(self, img_torch):

        # 最后一维度翻转
        trans = transforms.RandomHorizontalFlip(p=1)

        # img_torch=torch.from_numpy(np.array(img_torch))
        dim = -1
        if isinstance(img_torch,tuple):
            output=list(img_torch)
            for i in range(len(img_torch)):
               output[i]=trans(img_torch[i]).contiguous()
        else:
            dim = -1
            indices = [slice(None)] * img_torch.dim()
            indices[dim] = torch.arange(img_torch.size(dim) - 1, -1, -1,
                                        dtype=torch.long, device=img_torch.device)
            output = img_torch[tuple(indices)]
        # output = F.interpolate(img_torch, scale_factor=0.5, mode='bilinear')
        return output

    def fusion_feat(self,x_orig1,x_orig2):
        x_orig = []
        for i in range(4):
            # 直接融合
            conv_1_1_name = self.conv_1_1_layer[i]
            conv_1_1 = getattr(self, conv_1_1_name)
            f = torch.cat([x_orig1[i], x_orig2[i]], dim=1)
            f = conv_1_1(f)

            # 相减融合
            sub = torch.abs(x_orig1[i] - x_orig2[i])
            f = torch.cat([sub, f], dim=1)
            conv_1_2_name = self.conv_1_2_layer[i]
            conv_1_2 = getattr(self, conv_1_2_name)
            f = conv_1_2(f)
            x_orig.append(f)
        return x_orig

    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        s_map = score_map
        for i in range(1):
            index=2-i
            x_orig[index] = torch.cat([x_orig[index], s_map], dim=1)
            up_name = self.up_layer[i]
            up = getattr(self, up_name)
            s_map=up(s_map)

        return text_embeddings, x_orig, score_map

    def forward_train(self,
                      img1,
                      img2,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x1,x2= self.extract_feat(img1,img2)
        # x_orig1 = [x1[i] for i in range(4)]
        # x_orig2 = [x2[i] for i in range(4)]


        text_embeddings1, x_orig1, score_map1 = self.after_extract_feat(x1)
        text_embeddings2, x_orig2, score_map2 = self.after_extract_feat(x2)
        # score_map=torch.abs(score_map1-score_map2)

        # x_orig=self.fusion_feat(x_orig1,x_orig2)
        x_orig = []

        x_orig1 = list(x_orig1)

        for i in range(4):
            cross_name = self.cross_layer[i]
            cross_modul = getattr(self, cross_name)
            x_orig1[i] = cross_modul(text_embeddings1, x_orig1[i])
        x_orig1 = tuple(x_orig1)


        x_orig2 = list(x_orig2)
        for i in range(4):
            cross_name = self.cross_layer[i]
            cross_modul = getattr(self, cross_name)
            x_orig2[i] = cross_modul(text_embeddings2, x_orig2[i])
        x_orig2 = tuple(x_orig2)


        for i in range(4):
            # 直接融合
            conv_1_1_name = self.conv_1_1_layer[i]
            conv_1_1 = getattr(self, conv_1_1_name)
            f = torch.cat([x_orig1[i], x_orig2[i]], dim=1)
            f = conv_1_1(f)

            # 相减融合
            sub = torch.abs(x_orig1[i] - x_orig2[i])
            f = torch.cat([sub, f], dim=1)
            conv_1_2_name = self.conv_1_2_layer[i]
            conv_1_2 = getattr(self, conv_1_2_name)
            f = conv_1_2(f)
            x_orig.append(f)

        if self.with_neck:
            _x_orig=self.neck(x_orig)

        affine_x_neck = self.affine_transform(_x_orig)
        outs = self.bbox_head(_x_orig)
        ins_pred, cate_pred, levelset_feats = outs
        ins_pred2, cate_pred2, levelset_feats2 = self.bbox_head(affine_x_neck)
        fpn_feature = (ins_pred, ins_pred2)


        mask_and_score_list = []
        for i in range(len(gt_bboxes)):
            mask = np.zeros((256, 256), dtype=np.uint8)
            for j in range(gt_bboxes[i].size()[0]):
                X1, Y1, X2, Y2 = gt_bboxes[i][j].to('cpu')
                polygon = np.array([[X1, Y1], [X2, Y1], [X2, Y2], [X1, Y2]], np.int32)  # 坐标为顺时针方向
                cv2.fillConvexPoly(mask, polygon, (1, 1, 1))
            mask = torch.Tensor(mask).unsqueeze(0).to(gt_bboxes[0].device)
            mask_and_score_list.append(mask)

        mask_results=torch.cat(mask_and_score_list,dim=0)

        ###loss是否也需要2个图片？?
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img1 - img2, img_metas, self.train_cfg, fpn_feature,
                                  x1, x2, score_map1, score_map2, mask_results)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def visulize_all_channel_into_one(self,feature_map, i):
        '''热力图可视化'''
        output = feature_map
        output = output.data.squeeze()
        output = output.cpu().numpy()
        # print(output.shape)
        # output = np.mean(output, axis=0)
        # print(output.shape)
        height, width = 8, 8
        times = height / float(width)
        plt.rcParams["figure.figsize"] = (1, times)
        plt.axis('off')
        plt.imshow(output, cmap='jet', interpolation='bilinear')
        # plt.savefig('heatmap/heat{}.png'.format(i+1), dpi=3 * height)
        # plt.savefig('heatmap/heat{}.png'.format(i + 1), dpi=256, bbox_inches='tight', pad_inches=0)
        plt.savefig(i, dpi=256, bbox_inches='tight', pad_inches=0)
        plt.close()

    def simple_test(self, img1,img2, img_meta, rescale=False):
        x1,x2 = self.extract_feat(img1, img2)
        # x_orig1 = [x1[i] for i in range(4)]
        # x_orig2 = [x2[i] for i in range(4)]

        text_embeddings1, x_orig1, score_map1 = self.after_extract_feat(x1)
        text_embeddings2, x_orig2, score_map2 = self.after_extract_feat(x2)

        x_orig1 = list(x_orig1)
        for i in range(4):
            cross_name = self.cross_layer[i]
            cross_modul = getattr(self, cross_name)
            x_orig1[i] = cross_modul(text_embeddings1, x_orig1[i])
        x_orig1 = tuple(x_orig1)

        x_orig2 = list(x_orig2)
        for i in range(4):
            cross_name = self.cross_layer[i]
            cross_modul = getattr(self, cross_name)
            x_orig2[i] = cross_modul(text_embeddings2, x_orig2[i])
        x_orig2 = tuple(x_orig2)

        x_orig = []
        for i in range(4):
            # 直接融合
            conv_1_1_name = self.conv_1_1_layer[i]
            conv_1_1 = getattr(self, conv_1_1_name)
            f = torch.cat([x_orig1[i], x_orig2[i]], dim=1)
            f = conv_1_1(f)

            # 相减融合
            sub = torch.abs(x_orig1[i] - x_orig2[i])
            f = torch.cat([sub, f], dim=1)
            conv_1_2_name = self.conv_1_2_layer[i]
            conv_1_2 = getattr(self, conv_1_2_name)
            f = conv_1_2(f)
            x_orig.append(f)


        #可视化 分数图
        # out_dir = 'work_dirs/heatmaps/'
        # if os.path.exists(out_dir)==False:
        #     os.makedirs(out_dir)
        # s1 = score_map1
        # s2 = score_map2
        # # print(s1.device,score_mask.device)
        # diff = torch.mean(torch.abs(s1 - s2), dim=1).unsqueeze(1)
        # diff = torch.sigmoid(diff)
        # diff = torch.cat([diff, 1 - diff], dim=1)
        # feature_vis=diff
        # channel_num = feature_vis.shape[1]
        # for i in range(channel_num):
        #     out_file = os.path.join(out_dir, os.path.splitext(img_meta[0]['ori_filename'])[0]+'_'+str(i) + os.path.splitext(img_meta[0]['ori_filename'])[1])
        #     self.visulize_all_channel_into_one(feature_vis[0][i], out_file)

        if self.with_neck:
            _x_orig = self.neck(x_orig)

        # _x_orig = list(_x_orig)
        # text_diff_embedding = torch.abs(text_embeddings1 - text_embeddings2)
        # for i in range(4):
        #     cross_name = self.cross_layer[i]
        #     cross_modul = getattr(self, cross_name)
        #     _x_orig[i] = cross_modul(text_diff_embedding, _x_orig[i])
        # _x_orig = tuple(_x_orig)


        outs = self.bbox_head(_x_orig,eval=True)

        seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        results_list = self.bbox_head.get_seg(*seg_inputs)
        format_results_list = []
        for results in results_list:
            format_results_list.append(self.format_results(results))
        return format_results_list


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




