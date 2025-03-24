import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')

def visulize_all_channel_into_one(feature_map, i):
    output = feature_map
    output = output.data.squeeze()
    # print(output.shape)
    output = output.cpu().numpy()
    # print(output.shape)
    # output = np.mean(output, axis=0)
    # print(output.shape)
    height, width = 256, 256
    times = height / float(width)
    plt.rcParams["figure.figsize"] = (1, times)
    plt.axis('off')
    plt.imshow(output, cmap='jet', interpolation='bilinear')
    # plt.savefig('heatmap/heat{}.png'.format(i+1), dpi=3 * height)
    plt.savefig('heatmap/heat{}.png'.format(i+1), dpi=256, bbox_inches='tight', pad_inches=0)
    plt.close()

trf = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(p=1),
    transforms.Resize(128),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
])

# image_path1 = 'dataset/SYSU-CD/test/time1/03482.png'
# image_path2 = 'dataset/SYSU-CD/test/time2/03482.png'
# masks_path = 'dataset/SYSU-CD/test/label/03482.png'
# image_path1 = 'dataset/LEVIR-CD-256/test/A/test_106.png'
# image_path2 = 'dataset/LEVIR-CD-256/test/B/test_106.png'
# masks_path = 'dataset/LEVIR-CD-256/test/label/test_106.png'
# image_path1 = 'dataset/LEVIR-CD-256/test/A/test_1122.png'
# image_path2 = 'dataset/LEVIR-CD-256/test/B/test_1122.png'
# masks_path = 'dataset/LEVIR-CD-256/test/label/test_1122.png'
# image_path2 = 'dataset/LEVIR-CD-256/test/A/test_17.png'
# image_path1 = 'dataset/LEVIR-CD-256/test/B/test_17.png'
# masks_path = 'dataset/LEVIR-CD-256/test/label/test_17.png'
# image_path2 = 'dataset/LEVIR-CD-256/test/A/test_1101.png'
# image_path1 = 'dataset/LEVIR-CD-256/test/B/test_1101.png'
# masks_path = 'dataset/LEVIR-CD-256/test/label/test_1101.png'
image_path1 = 'dataset/BCDD/test/A/7541.png'
image_path2 = 'dataset/BCDD/test/B/7541.png'
masks_path = 'dataset/BCDD/test/label/7541.png'

images1 = Image.open(image_path1)
images2 = Image.open(image_path2)
masks = Image.open(masks_path).convert('L')
images1.save('heatmap/t1.png')
images2.save('heatmap/t2.png')
masks.save('heatmap/label.png')
images1 = trf(images1).unsqueeze(0).cuda()
images2 = trf(images2).unsqueeze(0).cuda()

# model_load = "result/idea2/resnet50_cond_up_4_2/SYSU-CD/checkpoints/best_model_epoch088_f_score0.8079_iou0.6777.pth"
# model_load = "result/idea4/resnet50_cls/LEVIR-CD-256/checkpoints/best_model_epoch120_precision0.9180.pth"
# model_load = "result/EGFDFN/resnet_simple_dsc_CBAM_dfm_edge/LEVIR-CD-256/checkpoints/best_model_epoch048_f_score0.9054_iou0.8272.pth"
# model_load = "result/idea4/superpixel_saliency_1/LEVIR-CD-256/checkpoints/best_model_epoch004_f_score0.2189_iou0.1229.pth"
# model_load = "result/idea4/resnet50_cls/LEVIR-CD-256/checkpoints/best_model_epoch120_precision0.9180.pth"
# model_load = "result/idea4/resnet50_cls_2/LEVIR-CD-256/checkpoints/best_model_epoch098_precision0.9312.pth"
# model_load = "result/idea4_new/resnet50_cam/LEVIR-CD-256/checkpoints/best_model_epoch047_f_score0.6862_iou0.5223_cls_precision0.9717.pth"
# model_load = "result/idea4_new/resnet50_icr_largecam/LEVIR-CD-256/checkpoints/best_model_epoch069_f_score0.7247_iou0.5682_cls_precision0.9746.pth"
# model_load = "result/idea4_new/resnet50_icr_largecam_mask/LEVIR-CD-256/checkpoints/best_model_epoch111_f_score0.7556_iou0.6072_cls_precision0.9688.pth"
# model_load = "result/idea4_new/resnet50_icr_largecam/BCDD/checkpoints/best_model_epoch006_f_score0.7790_iou0.6379_cls_precision0.9725.pth"
model_load = "result/idea4_new/resnet50_cam/BCDD/checkpoints/best_model_epoch004_f_score0.6925_iou0.5296_cls_precision0.9685.pth"
net = torch.load(model_load)

net_input = torch.cat([images1.unsqueeze(0), images2.unsqueeze(0)], dim=0)
out = net(net_input)
# out = net(images1, images2)
feature_vis = out[0]
# feature_vis = out[19]
# feature_vis = out[0][0][1].unsqueeze(0).unsqueeze(0)
# feature_vis[feature_vis>0.95] = 1
# feature_vis[feature_vis<=0.95] = 0
print(feature_vis.shape)
# print(feature_vis)
# trans = transforms.RandomHorizontalFlip(p=1)
trans = transforms.Resize(256)
feature_vis = trans(feature_vis)
# print(feature_vis)
# feature_vis = F.softmax(feature_vis, dim=1)
# print(feature_vis)

channel_num = feature_vis.shape[1]
# print(channel_num)
print(feature_vis)

for i in tqdm(range(channel_num)):
    visulize_all_channel_into_one(feature_vis[0][i], i)

# out = out[0] + out[1] + out[2]
# _, pred = torch.max(out, dim=1)
# pred = pred.cpu().detach().numpy()
# pred = pred * 255.0
# vis = np.array(pred).transpose((1, 2, 0))
# cv2.imshow("vis", vis)
# cv2.waitKey(0)