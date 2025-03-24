import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
from utils.denseCRF import CRFs
from utils.numpy_metrics import binary_classification_metrics
import random

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
# model = 'resnet50_true_icr_feature_2_dualcls_1'
# model = 'resnet50_true_icr_feature_2_dualcls_3'
# model = 'resnet50_true_icr_feature_2_dualcls_0.05'
# model = 'resnet50_true_icr_feature_2_dualcls_reverse'
# model = 'resnet50_cam'
# model = 'resnet50_cam_0percent'
# model = 'resnet50_cam_10percent'
# model = 'resnet50_icr_largecam'
# model = 'resnet50_icr_largecam_mask'
# model = 'resnet50_icr_largecam_mask_2'
model = 'resnet50_icr_largecam_mask_0epoch'
# model = 'resnet50_icr_largecam_mask_CRF_only'
# model = 'resnet50_icr_largecam_mask_softmax'
# model = 'resnet50_icr_largecam_mask_softmax_hole'
# model = 'resnet50_icr_largecam_mask_2_softmax'
# model = 'resnet50_icr_largecam_affine_mask_1'
# model = 'resnet50_icr_largecam_affine_mask_1_1'
# model = 'resnet50_icr_largecam_affine_mask_2'
# model = 'resnet50_icr_largecam_affine_mask_2_1'
# model = 'SEAM'
# model = 'MJS_ICR_Net'

dataset = 'LEVIR-CD-256'
files = 'dataset/LEVIR-CD-256/test'
image_path1 = glob.glob(files + '/A/*')
image_path2 = glob.glob(files + '/B/*')
masks_path = glob.glob(files + '/label/*')
# net = torch.load('result/idea4_new/resnet50_cam/LEVIR-CD-256/checkpoints/best_model_epoch047_f_score0.6862_iou0.5223_cls_precision0.9717.pth')
# net = torch.load('result/idea4_new/resnet50_cam_0percent/LEVIR-CD-256/checkpoints/best_model_epoch022_f_score0.6404_iou0.4710_cls_precision0.9233.pth')
# net = torch.load('result/idea4_new/resnet50_cam_10percent/LEVIR-CD-256/checkpoints/best_model_epoch030_f_score0.6452_iou0.4762_cls_precision0.9673.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam/LEVIR-CD-256/checkpoints/best_model_epoch069_f_score0.7247_iou0.5682_cls_precision0.9746.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam_mask/LEVIR-CD-256/checkpoints/best_model_epoch111_f_score0.7556_iou0.6072_cls_precision0.9688.pth')
# net = torch.load('result/idea4_new/SEAM/LEVIR-CD-256/checkpoints/best_model_epoch022_f_score0.7287_iou0.5732_cls_precision0.9692.pth')
# net = torch.load('result/idea4_new/MJS_ICR_Net/LEVIR-CD-256/checkpoints/best_model_epoch014_f_score0.7530_iou0.6038_cls_precision0.9678.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam_affine_mask_1/LEVIR-CD-256/checkpoints/best_model_epoch152_f_score0.7520_iou0.6026_cls_precision0.9668.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam_affine_mask_2/LEVIR-CD-256/checkpoints/best_model_epoch146_f_score0.7509_iou0.6012_cls_precision0.9702.pth')
net = torch.load('result/idea4_new/resnet50_icr_largecam_mask/LEVIR-CD-256/checkpoints/best_model_epoch000_f_score0.4775_iou0.3136_cls_precision0.9170.pth')

# dataset = 'BCDD'
# files = 'dataset/BCDD/test'
# image_path1 = glob.glob(files + '/A/*')
# image_path2 = glob.glob(files + '/B/*')
# masks_path = glob.glob(files + '/label/*')
# # net = torch.load('result/idea4_new/resnet50_cam/BCDD/checkpoints/best_model_epoch004_f_score0.6925_iou0.5296_cls_precision0.9685.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam/BCDD/checkpoints/best_model_epoch006_f_score0.7790_iou0.6379_cls_precision0.9725.pth')
# # net = torch.load('result/idea4_new/resnet50_icr_largecam_mask/BCDD/checkpoints/best_model_epoch004_f_score0.7430_iou0.5911_cls_precision0.9502.pth')
# # net = torch.load('result/idea4_new/SEAM/BCDD/checkpoints/best_model_epoch003_f_score0.7430_iou0.5910_cls_precision0.9541.pth')
# # net = torch.load('result/idea4_new/MJS_ICR_Net/BCDD/checkpoints/best_model_epoch009_f_score0.7681_iou0.6235_cls_precision0.9397.pth')

# dataset = 'CD_GZ'
# files = 'dataset/CD_GZ/test'
# image_path1 = glob.glob(files + '/A/*')
# image_path2 = glob.glob(files + '/B/*')
# masks_path = glob.glob(files + '/label/*')
# # net = torch.load('result/idea4_new/resnet50_cam/CD_GZ/checkpoints/best_model_epoch006_f_score0.6846_iou0.5205_cls_precision0.9249.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam/CD_GZ/checkpoints/best_model_epoch079_f_score0.7424_iou0.5903_cls_precision0.9377.pth')
# # net = torch.load('result/idea4_new/resnet50_icr_largecam_mask/CD_GZ/checkpoints/best_model_epoch042_f_score0.7186_iou0.5608_cls_precision0.9121.pth')
# # net = torch.load('result/idea4_new/SEAM/CD_GZ/checkpoints/best_model_epoch004_f_score0.7168_iou0.5586_cls_precision0.9026.pth')
# # net = torch.load('result/idea4_new/MJS_ICR_Net/CD_GZ/checkpoints/best_model_epoch039_f_score0.7042_iou0.5435_cls_precision0.9265.pth')

# dataset = 'EGY_BCD'
# files = 'dataset/EGY_BCD/test'
# image_path1 = glob.glob(files + '/A/*')
# image_path2 = glob.glob(files + '/B/*')
# masks_path = glob.glob(files + '/label/*')
# # net = torch.load('result/idea4_new/resnet50_cam/EGY_BCD/checkpoints/best_model_epoch022_f_score0.5528_iou0.3820_cls_precision0.9525.pth')
# net = torch.load('result/idea4_new/resnet50_icr_largecam/EGY_BCD/checkpoints/best_model_epoch014_f_score0.5521_iou0.3813_cls_precision0.9541.pth')

# dataset = 'SYSU-CD'
# files = 'dataset/SYSU-CD/test'
# image_path1 = glob.glob(files + '/time1/*')
# image_path2 = glob.glob(files + '/time2/*')
# masks_path = glob.glob(files + '/label/*')
# net = torch.load('result/idea2/resnet50_cond_up_4_2/SYSU-CD/checkpoints/best_model_epoch088_f_score0.8079_iou0.6777.pth')

tp_all = 0.0
fp_all = 0.0
fn_all = 0.0

tp_crf_all = 0.0
fp_crf_all = 0.0
fn_crf_all = 0.0

def write_color_img(pred, label, out_path):
    if pred.shape[-1] == 3 or pred.shape[-1] == 1:
        pred = pred[:, :, 0]
    if label.shape[-1] == 3 or label.shape[-1] == 1:
        label = label[:, :, 0]
    tp_mask = np.logical_and(pred == 255, label == 255)
    fp_mask = np.logical_and(pred == 255, label == 0)
    fn_mask = np.logical_and(pred == 0, label == 255)
    tn_mask = np.logical_and(pred == 0, label == 0)

    result_image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    result_image[tp_mask] = [255, 255, 255]  # White
    result_image[fp_mask] = [0, 0, 255]  # Red
    result_image[fn_mask] = [255, 0, 0]  # Blue
    result_image[tn_mask] = [0, 0, 0]  # Black

    cv2.imwrite(out_path, result_image)

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

    images1 = trf(images1).unsqueeze(0).unsqueeze(0).to(device)
    images2 = trf(images2).unsqueeze(0).unsqueeze(0).to(device)
    image_input = torch.cat([images1, images2], dim=0)
    # cam, cls = net(image_input)
    cam1, cam2, cam3, cam1_, cam2_, cam3_, \
        out1, out2, out3, out1_, out2_, out3_, \
        x1_e1, x2_e1, x3_e1, x4_e1, x1_e2, x2_e2, x3_e2, x4_e2, \
        cls1, cls2, cls3, cls1_, cls2_, cls3_ = net(image_input)
    # cam, cam_p, cam_, cam_p_, cls, cls_ = net(image_input)
    # cam2, cam3, cam4, cam2_, cam3_, cam4_, cls2, cls3, cls4, cls2_, cls3_, cls4_ = net(image_input)

    # cam = cam
    cam = (cam1 + cam2 + cam3) / 3.0
    # cam = cam
    # cam = (cam2 + F.interpolate(cam3, scale_factor=2, mode='bilinear', align_corners=True) + F.interpolate(cam4, scale_factor=2, mode='bilinear', align_corners=True)) / 3.0
    cam = F.interpolate(cam, size=256, mode='bilinear', align_corners=True)
    # cls = F.softmax(cls, dim=1)
    cls = F.softmax(cls3, dim=1)
    # cls = F.softmax(cls, dim=1)
    # cls = F.softmax(cls4, dim=1)
    cls = cls[0]
    if cls[0] > 0.9999:
    # if cls[0] > 0.99999:
    # if cls[0] > 0.5:
        pred = torch.zeros_like(cam[0][1].unsqueeze(0)).float()
    else:
        pred = cam[0][1].unsqueeze(0)
        threshold = 0.95
        # threshold = 0.3
        # threshold = 0.65
        # threshold = 0.75
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
    use_CRF = True

    # print(pred.shape)
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

    predicted_image = cv2.imread(vis_path)
    mask = masks.cpu().detach().numpy()
    mask = mask * 255.0
    mask = np.array(mask).transpose((1, 2, 0))

    # CRF
    if use_CRF:
        sim1 = F.cosine_similarity(out1,x1_e1).sum().item()
        sim2 = F.cosine_similarity(out1,x1_e2).sum().item()
        if sim1 > sim2:
            original_image = cv2.imread(image_path1[i])
        else:
            original_image = cv2.imread(image_path2[i])

        # original_image = cv2.imread(image_path2[i])

        feature1 = torch.sigmoid(x4_e1[0][0])
        feature2 = torch.sigmoid(x4_e2[0][0])
        cam = F.interpolate(cam, size=32, mode='bilinear', align_corners=True)[0][1]
        sim1 = cam * feature1
        sim2 = cam * feature2
        if sim1.sum() > sim2.sum():
            original_image = cv2.imread(image_path1[i])
        else:
            original_image = cv2.imread(image_path2[i])

        # rand = random.random()
        # if rand < 0.5:
        #     original_image = cv2.imread(image_path1[i])
        # else:
        #     original_image = cv2.imread(image_path2[i])

        if not np.all(predicted_image == 0):
            pred_crf = CRFs(original_image, predicted_image, sxy1=1, sxy2=10, srgb=10)
        else:
            pred_crf = predicted_image
        # print(pred_crf.shape)
        vis_crf_root = 'test_image_all/{}/{}/vis_crf'.format(dataset, model)
        if not os.path.exists(vis_crf_root):
            os.makedirs(vis_crf_root)
        TP, FP, FN = binary_classification_metrics(pred_crf[:,:,0], mask[:,:,0])
        precision_crf = TP / (TP + FP) if TP + FP > 0 else 0
        recall_crf = TP / (TP + FN) if TP + FN > 0 else 0
        f1_crf = 2 * (precision_crf * recall_crf) / (precision_crf + recall_crf) if precision_crf + recall_crf > 0 else 0
        f1_crf = round(f1_crf*100, 2)
        tp_crf_all += TP
        fp_crf_all += FP
        fn_crf_all += FN
        vis_crf_path = '{}/{}_{}_{}.png'.format(vis_crf_root, index, model, f1_crf)
        cv2.imwrite(vis_crf_path, pred_crf)

    # color
    vis_color_root = 'test_image_all/{}/{}/vis_color'.format(dataset, model)
    if not os.path.exists(vis_color_root):
        os.makedirs(vis_color_root)
    vis_color_path = '{}/{}_{}_{}.png'.format(vis_color_root, index, model, f1)
    write_color_img(predicted_image, mask, vis_color_path)
    if use_CRF:
        vis_crf_color_root = 'test_image_all/{}/{}/vis_crf_color'.format(dataset, model)
        if not os.path.exists(vis_crf_color_root):
            os.makedirs(vis_crf_color_root)
        vis_crf_color_path = '{}/{}_{}_{}.png'.format(vis_crf_color_root, index, model, f1_crf)
        write_color_img(pred_crf, mask, vis_crf_color_path)

    # break

precision = tp_all / (tp_all + fp_all)
recall = tp_all / (tp_all + fn_all)
f1 = 2 * (tp_all / (tp_all + fp_all)) * (tp_all / (tp_all + fn_all)) / (
            (tp_all / (tp_all + fp_all)) + (tp_all / (tp_all + fn_all)))
iou = tp_all / (tp_all + fn_all + fp_all)
print('pre:{}, recall:{}, f1:{}, iou:{}'.format(precision, recall, f1, iou))
os.rename(vis_root, vis_root + '_f1_{:.2f}_iou_{:.2f}'.format(f1*100, iou*100))

if use_CRF:
    print('CRF:')
    precision = tp_crf_all / (tp_crf_all + fp_crf_all)
    recall = tp_crf_all / (tp_crf_all + fn_crf_all)
    f1 = 2 * (tp_crf_all / (tp_crf_all + fp_crf_all)) * (tp_crf_all / (tp_crf_all + fn_crf_all)) / (
                (tp_crf_all / (tp_crf_all + fp_crf_all)) + (tp_crf_all / (tp_crf_all + fn_crf_all)))
    iou = tp_crf_all / (tp_crf_all + fn_crf_all + fp_crf_all)
    print('pre:{}, recall:{}, f1:{}, iou:{}'.format(precision, recall, f1, iou))
    os.rename(vis_crf_root, vis_crf_root + '_f1_{:.2f}_iou_{:.2f}'.format(f1*100, iou*100))