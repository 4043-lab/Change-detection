import os
import numpy as np
from PIL import Image
import glob
import collections
import torch
import torch.nn.functional as F
from torchvision import transforms
import datetime
import os
import argparse
from utils.logsetting import get_log
from utils.metrics import Metrics
import cv2
# import models
# from model import model_test,final_SE_dual_dfm_CBAM
from tqdm import tqdm

trf = transforms.Compose([
    transforms.ToTensor(),
])
device = 'cuda'
# model = 'FC-EF'
# model = 'FC_siam_conc'
# model = 'FC_siam_diff'
# model = 'BIT'
# model = 'SNUNet'
# model = 'P2V'
# model = 'DMINet'
# model = 'ours'
# model = 'resnet50_cond_4'
# model = 'resnet50_cond_up_4'
# model = 'resnet50_true_icr_feature_2'
# model = 'resnet50_true_icr_feature_2_dualcls_3'
# model = 'resnet50_true_icr_feature_2_dualcls_0.05'
# model = 'resnet50_true_icr_feature_2_dualcls_reverse'
model = 'resnet_simple_dsc_CBAM_dfm_2_edge_2'

# dataset = 'LEVIR-CD-256'
# files = 'dataset/LEVIR-CD-256/test'
# image_path1 = glob.glob(files + '/A/*')
# image_path2 = glob.glob(files + '/B/*')
# masks_path = glob.glob(files + '/label/*')
# # net = torch.load('result/comparison/FC_EF/LEVIR-CD-256/checkpoints/best_model_epoch097_f_score0.8356_iou0.7176.pth')
# # net = torch.load('result/comparison/FC_siam_conc/LEVIR-CD-256/checkpoints/best_model_epoch099_f_score0.8705_iou0.7706.pth')
# # net = torch.load('result/comparison/FC_siam_diff/LEVIR-CD-256/checkpoints/best_model_epoch098_f_score0.8681_iou0.7669.pth')
# # net = torch.load('result/comparison/BIT/LEVIR-CD-256/checkpoints/best_model_epoch039_f_score0.8895_iou0.8009.pth')
# # net = torch.load('result/comparison/SNUNet/LEVIR-CD-256/checkpoints/best_model_epoch072_f_score0.9028_iou0.8228.pth')
# # net = torch.load('result/comparison/P2V/LEVIR-CD-256/checkpoints/best_model_epoch043_f_score0.9028_iou0.8229.pth')
# # net = torch.load('result/comparison/DMINet/LEVIR-CD-256/checkpoints/best_model_epoch068_f_score0.8980_iou0.8149.pth')
# # net = torch.load('result/idea2/final_cond_ConvLSTM_AW_5/LEVIR-CD-256/checkpoints/best_model_epoch003_f_score0.9135_iou0.8407.pth')
# # net = torch.load('result/idea2/ablation/resnet50_cond_k4/LEVIR-CD-256/checkpoints/best_model_epoch049_f_score0.9110_iou0.8366.pth')
# # net = torch.load('result/idea2/resnet50_cond_up_4/LEVIR-CD-256/checkpoints/best_model_epoch069_f_score0.9122_iou0.8386.pth')
# # net = torch.load('result/idea4/resnet50_true_icr_feature_2_1/LEVIR-CD-256/checkpoints/best_model_epoch010_f_score0.6603_iou0.4928_cls_precision0.9185.pth')
# # net = torch.load('result/idea4/resnet50_true_icr_feature_2_dualcls_3/LEVIR-CD-256/checkpoints/best_model_epoch063_f_score0.7328_iou0.5782_cls_precision0.9741.pth')
# # net = torch.load('result/idea4/resnet50_true_icr_feature_2_dualcls_3/LEVIR-CD-256/checkpoints/best_model_epoch120_f_score0.7374_iou0.5840_cls_precision0.9644.pth')
# net = torch.load('result/EGFDFN/resnet_simple_dsc_CBAM_dfm_2_edge_2/LEVIR-CD-256/checkpoints/best_model_epoch051_f_score0.9114_iou0.8372.pth')

# dataset = 'BCDD'
# files = 'dataset/BCDD/test'
# image_path1 = glob.glob(files + '/A/*')
# image_path2 = glob.glob(files + '/B/*')
# masks_path = glob.glob(files + '/label/*')
# # net = torch.load('result/comparison/FC_EF/BCDD/checkpoints/best_model_epoch097_f_score0.8229_iou0.6991.pth')
# # net = torch.load('result/comparison/FC_siam_conc/BCDD/checkpoints/best_model_epoch160_f_score0.8812_iou0.7876.pth')
# # net = torch.load('result/comparison/FC_siam_diff/BCDD/checkpoints/best_model_epoch155_f_score0.8630_iou0.7590.pth')
# # net = torch.load('result/comparison/BIT/BCDD/checkpoints/best_model_epoch014_f_score0.9085_iou0.8323.pth')
# # net = torch.load('result/comparison/SNUNet/BCDD/checkpoints/best_model_epoch141_f_score0.9202_iou0.8523.pth')
# # net = torch.load('result/comparison/P2V/BCDD/checkpoints/best_model_epoch242_f_score0.9284_iou0.8663.pth')
# # net = torch.load('result/comparison/DMINet/BCDD/checkpoints/best_model_epoch088_f_score0.9288_iou0.8670.pth')
# # net = torch.load('result/idea2/final_cond_ConvLSTM_AW_5/BCDD/checkpoints/best_model_epoch000_f_score0.9371_iou0.8817.pth')
# # net = torch.load('result/idea4/resnet50_true_icr_feature_2_dualcls_3/BCDD/checkpoints/best_model_epoch045_f_score0.7432_iou0.5914_cls_precision0.9581.pth')
# net = torch.load('result/EGFDFN/resnet_simple_dsc_CBAM_dfm_2_edge_2/BCDD/checkpoints/best_model_epoch259_f_score0.9211_iou0.8538.pth')

dataset = 'SYSU-CD'
files = 'dataset/SYSU-CD/test'
image_path1 = glob.glob(files + '/time1/*')
image_path2 = glob.glob(files + '/time2/*')
masks_path = glob.glob(files + '/label/*')
# net = torch.load('result/comparison/FC_EF/SYSU-CD/checkpoints/best_model_epoch072_f_score0.7262_iou0.5701.pth')
# net = torch.load('result/comparison/FC_siam_conc/SYSU-CD/checkpoints/best_model_epoch070_f_score0.7169_iou0.5587.pth')
# net = torch.load('result/comparison/FC_siam_diff/SYSU-CD/checkpoints/best_model_epoch064_f_score0.6851_iou0.5210.pth')
# net = torch.load('result/comparison/BIT/SYSU-CD/checkpoints/best_model_epoch042_f_score0.7795_iou0.6387.pth')
# net = torch.load('result/comparison/SNUNet/SYSU-CD/checkpoints/best_model_epoch046_f_score0.7841_iou0.6449.pth')
# net = torch.load('result/comparison/P2V/SYSU-CD/checkpoints/best_model_epoch018_f_score0.7921_iou0.6557.pth')
# net = torch.load('result/comparison/DMINet/SYSU-CD/checkpoints/best_model_epoch024_f_score0.8058_iou0.6747.pth')
# net = torch.load('result/idea2/final_cond_ConvLSTM_AW_5/SYSU-CD/checkpoints/best_model_epoch021_f_score0.8142_iou0.6867.pth')
# net = torch.load('result/idea2/ablation/resnet50_cond_k4/SYSU-CD/checkpoints/best_model_epoch093_f_score0.8021_iou0.6696.pth')
# net = torch.load('result/idea2/resnet50_cond_up_4_2/SYSU-CD/checkpoints/best_model_epoch088_f_score0.8079_iou0.6777.pth')
net = torch.load('result/EGFDFN/resnet_simple_dsc_CBAM_dfm_2_edge_2/SYSU-CD/checkpoints/best_model_epoch017_f_score0.7957_iou0.6607.pth')

# dataset = 'CD_GZ'
# files = 'dataset/CD_GZ/test'
# image_path1 = glob.glob(files + '/A/*')
# image_path2 = glob.glob(files + '/B/*')
# masks_path = glob.glob(files + '/label/*')
# # net = torch.load('result/comparison/FC_EF/CD_GZ/checkpoints/best_model_epoch338_f_score0.8051_iou0.6737.pth')
# # net = torch.load('result/comparison/FC_siam_conc/CD_GZ/checkpoints/best_model_epoch354_f_score0.8185_iou0.6928.pth')
# net = torch.load('result/comparison/FC_siam_diff/CD_GZ/checkpoints/best_model_epoch525_f_score0.8241_iou0.7009.pth')

tp_all = 0.0
fp_all = 0.0
fn_all = 0.0

for i in tqdm(range(len(masks_path))):
# for i in range(len(masks_path)):
    if dataset == 'CD_GZ':
        index = os.path.basename(image_path1[i]).split('.')[0]
    else:
        index = os.path.basename(image_path1[i]).split('.')[0].split('_')[-1]
    masks = Image.open(masks_path[i]).convert('L')
    masks = trf(masks)
    masks = (masks > 0).type(torch.LongTensor).to(device)
    images1 = Image.open(image_path1[i])
    images2 = Image.open(image_path2[i])

    # images1 = trf(images1).unsqueeze(0).unsqueeze(0).to(device)
    # images2 = trf(images2).unsqueeze(0).unsqueeze(0).to(device)
    # image_input = torch.cat([images1, images2], dim=0)
    # outs = net(image_input)

    images1 = trf(images1).unsqueeze(0).to(device)
    images2 = trf(images2).unsqueeze(0).to(device)
    outs = net(images1, images2)

    if model == 'DMINet':
        out = outs[0] + outs[1]
    elif model == 'ours':
        out = outs[0] + outs[1] + outs[2]
    elif model == 'resnet50_true_icr_feature_2_dualcls_3':
        out = outs[0]
        out = F.interpolate(out, size=256, mode='bilinear', align_corners=True)
    else:
        if isinstance(outs, (list, tuple)):
            out = outs[-1]
        else:
            out = outs

    _, pred = torch.max(out, dim=1)
    tp = masks * pred
    fp = pred * (1 - masks)
    fn = masks * (1 - pred)
    tp_sum = torch.sum(tp)
    fp_sum = torch.sum(fp)
    fn_sum = torch.sum(fn)
    tp_all += tp_sum.item()
    fp_all += fp_sum.item()
    fn_all += fn_sum.item()
    f1 = 2*(tp_sum/(tp_sum+fp_sum))*(tp_sum/(tp_sum+fn_sum)) / ((tp_sum/(tp_sum+fp_sum))+(tp_sum/(tp_sum+fn_sum)))
    f1 = 100*f1.item()
    f1 = round(f1, 2)

    pred = pred.cpu().detach().numpy()
    pred = pred * 255.0
    vis = np.array(pred).transpose((1, 2, 0))
    vis_root = 'test_image_all/{}/{}/vis'.format(dataset, model)
    if not os.path.exists(vis_root):
        os.makedirs(vis_root)
    vis_path = '{}/{}_{}_{}.png'.format(vis_root, index, model, f1)
    cv2.imwrite(vis_path, vis)

    one = torch.ones([1, 256, 256]) * 255.0
    zero = torch.zeros([1, 256, 256])
    c_blue = torch.cat([one, zero, zero], dim=0)
    # c_green = torch.cat([zero,one,zero],dim=0)
    c_red = torch.cat([zero, zero, one], dim=0)
    c_white = torch.cat([one, one, one], dim=0)
    color_image = c_red * fp.cpu() + c_blue * fn.cpu() + c_white * tp.cpu()
    color_image = np.array(color_image).transpose((1, 2, 0))
    vis_color_root = 'test_image_all/{}/{}/vis_color'.format(dataset, model)
    if not os.path.exists(vis_color_root):
        os.makedirs(vis_color_root)
    vis_color_path = '{}/{}_{}_{}.png'.format(vis_color_root, index, model, f1)
    cv2.imwrite(vis_color_path, color_image)

precision = tp_all / (tp_all + fp_all)
recall = tp_all / (tp_all + fn_all)
f1 = 2 * (tp_all / (tp_all + fp_all)) * (tp_all / (tp_all + fn_all)) / (
            (tp_all / (tp_all + fp_all)) + (tp_all / (tp_all + fn_all)))
iou = tp_all / (tp_all + fn_all + fp_all)
print('pre:{}, recall:{}, f1:{}, iou:{}'.format(precision, recall, f1, iou))
os.rename(vis_root, vis_root + '_pre_{:.2f}_rec_{:.2f}_f1_{:.2f}_iou_{:.2f}'.format(precision*100, recall*100, f1*100, iou*100))