"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import seg_metrics
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os
import copy
from tqdm import tqdm
from utils.metrics_cls import Metrics_cls
from utils.metrics import Metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def train_model(model, dataloaders, criterion, criterion_weight, optimizer, sc_plt, num_epochs=100, model_name='',
                model_info='', model_type='', dataset_name='', begin_epoch=0, best_F1=0.0, out_cam=False,
                change_threshold = 0.0):
    model_root = 'result/{}/{}/{}'.format(model_type, model_name, dataset_name)
    # path_checkpoints = 'result/{}/{}/checkpoints'.format(model_type, model_name)
    # path_figure = 'result/{}/{}/figure'.format(model_type, model_name)
    # path_note = 'result/{}/{}/note.txt'.format(model_type, model_name)
    path_checkpoints = model_root + '/checkpoints'
    path_figure = model_root + '/figure'
    path_note = model_root + '/note.txt'

    if not os.path.exists(model_root):
        os.makedirs(model_root)
    if not os.path.exists(path_checkpoints):
        os.mkdir(path_checkpoints)
    if not os.path.exists(path_figure):
        os.mkdir(path_figure)
    with open(path_note, 'w') as f:
        f.write(model_info)


    val_acc = []
    val_loss = []
    train_loss = []

    best_F1 = best_F1
    for epoch in range(begin_epoch, num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        metrics_cls = Metrics_cls()
        # metrics = [Metrics()] * 20
        metrics = []
        for i in range(20):
            tmp = Metrics()
            metrics.append(tmp)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        # for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss_cls = 0.0
            running_loss_cri = 0.0
            running_loss_crf = 0.0
            running_loss_seg = 0.0
            running_loss_ecr = 0.0
            running_loss_er = 0.0
            running_loss = 0.0

            # Iterate over data.
            for sample in tqdm(dataloaders[phase]):
            # for sample in dataloaders[phase]:
                t1 = sample['reference'].unsqueeze(0).cuda() #image1
                t2 = sample['test'].unsqueeze(0).cuda() #image2
                t = torch.cat([t1, t2], dim=0)

                labels = (sample['label'] > 0).squeeze(1).type(torch.LongTensor).cuda()
                # print(labels)
                # print("label形状", labels.shape)
                labels_cls = torch.sum(labels, dim=(1, 2))
                # print(labels_cls)
                # labels_cls[labels_cls>0] = 1  # [B] 分类标签
                threshold_true = 65536 * change_threshold
                labels_cls[labels_cls <= threshold_true] = 0  # [B] 分类标签
                labels_cls[labels_cls > threshold_true] = 1  # [B] 分类标签
                # print(labels_cls)
                # print(labels)

                # # # labels_zero = torch.zeros_like(labels)
                # labels_zero_32 = torch.zeros([1, 1, 32, 32]).cuda()
                # labels_one_32 = torch.ones([1, 1, 32, 32]).cuda()
                # labels_seg_32 = torch.cat([labels_zero_32, labels_one_32], dim=1).float()
                # labels_zero_64 = torch.zeros([1, 1, 64, 64]).cuda()
                # labels_one_64 = torch.ones([1, 1, 64, 64]).cuda()
                # labels_seg_64 = torch.cat([labels_zero_64, labels_one_64], dim=1).float()
                # # print(labels_seg.shape)
                # # print(labels_seg)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  # train 计算梯度，val的时候不计算

                    # out_cls = model(t)[-1]  # [B, 2]
                    # # print(out_cls.shape)
                    #
                    # loss = criterion(out_cls, labels)
                    # # print(loss)

                    out = model(t)

                    loss_cls = torch.tensor(0.0).cuda()  # 分类损失
                    loss_cri = torch.tensor(0.0).cuda()  # 多尺度一致性损失
                    loss_crf = torch.tensor(0.0).cuda()  # 仿射变换一致性损失
                    loss_seg = torch.tensor(0.0).cuda()  # 未变化全监督损失

                    # SEAM
                    loss_ecr = torch.tensor(0.0).cuda()
                    loss_er = torch.tensor(0.0).cuda()

                    # # cri_loss
                    # cam2 = out[0]
                    # cam2 = F.interpolate(cam2, scale_factor=0.5, mode='bilinear', align_corners=True)
                    # cam3 = out[1]
                    # cam4 = out[2]
                    # loss_cri += criterion[0](cam2, cam3)
                    # loss_cri += criterion[0](cam2, cam4)
                    # loss_cri += criterion[0](cam3, cam4)

                    # # crf_loss
                    # for i in range(0, 3):
                    # # for i in range(6, 9):
                    #     loss_crf += criterion[0](out[i], out[i+3])

                    # cls_loss
                    # for i in range(-6, 0):
                    #     loss_cls += criterion[1](out[i], labels_cls)
                    for i in range(-1, 0):
                        loss_cls += criterion[1](out[i], labels_cls)

                    # # seg_loss
                    # bs = t1.shape[1]
                    # for b in range(bs):
                    #     if(labels_cls[b] == 0):
                    #         # loss_seg += criterion[2](out[0][b].unsqueeze(0), labels_zero)
                    #         # loss_seg += criterion[2](out[1][b].unsqueeze(0), labels_zero)
                    #         loss_seg += criterion[0](out[0][b].unsqueeze(0), labels_seg_64)
                    #         loss_seg += criterion[0](out[1][b].unsqueeze(0), labels_seg_32)
                    #         loss_seg += criterion[0](out[2][b].unsqueeze(0), labels_seg_32)
                    #     # print(b, loss_seg)

                    # # SEAM
                    # # loss_ecr
                    # loss_ecr += criterion[0](out[0], out[3])
                    # loss_ecr += criterion[0](out[1], out[2])
                    #
                    # # loss_er
                    # loss_er += criterion[0](out[0], out[2])

                    loss = loss_cls
                    # loss = loss_cls + loss_cri
                    # loss = loss_cls + loss_crf
                    # loss = loss_cls + loss_cri + loss_crf
                    # loss = loss_cls + loss_cri + loss_crf + loss_seg
                    # loss = loss_cls + loss_crf + loss_seg
                    # loss = loss_cls + loss_er
                    # loss = loss_cls + loss_ecr + loss_er

                    # Calculate metric during evaluation
                    if phase == 'val':
                        # cls = out[-2]
                        # cls = out[-4]
                        cls = out[-1]

                        metrics_cls.add(labels_cls, cls)
                        cam = out[0]
                        # cam = out[1]
                        # cam2 = out[0]
                        # cam3 = F.interpolate(out[1], scale_factor=2, mode='bilinear', align_corners=True)
                        # cam4 = F.interpolate(out[2], scale_factor=2, mode='bilinear', align_corners=True)
                        # cam = (cam2 + cam3 + cam4) / 3.0
                        # cam = (out[0] + out[1] + out[2]) / 3.0
                        cam = F.interpolate(cam, size=256, mode='bilinear', align_corners=True)

                        for i in range(cam.shape[0]):
                            if cls[i][0] > cls[i][1]:
                                for j in range(20):
                                    metrics[j].add(labels[i], torch.zeros_like(cam[0]))
                            else:
                                cam_single = cam[i][1].unsqueeze(0)
                                # C, H, W = cam_single.shape
                                # cam_single = F.relu(cam_single)
                                # max_v = torch.max(cam_single.view(C, -1), dim=1)[0].view(C, 1, 1)
                                # min_v = torch.min(cam_single.view(C, -1), dim=1)[0].view(C, 1, 1)
                                # cam_single = F.relu(cam_single - min_v - 1e-5) / (max_v - min_v + 1e-5)

                                # print(cam_single)
                                for j in range(20):
                                    threshold = j * 0.05
                                    cam_clone = cam_single.clone()
                                    cam_clone[cam_clone <= threshold] = 0.0
                                    cam_clone[cam_clone > threshold] = 1.0
                                    metrics[j].add(labels[i], cam_clone)
                                    # print(i, j*0.05, metrics[0].get_f_score())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss_cls += loss_cls.item()
                running_loss_cri += loss_cri.item()
                running_loss_crf += loss_crf.item()
                running_loss_seg += loss_seg.item()
                running_loss_ecr += loss_ecr.item()
                running_loss_er += loss_er.item()
                running_loss += loss.item()

            epoch_loss_cls = running_loss_cls / len(dataloaders[phase].dataset)
            epoch_loss_cri = running_loss_cri / len(dataloaders[phase].dataset)
            epoch_loss_crf = running_loss_crf / len(dataloaders[phase].dataset)
            epoch_loss_seg = running_loss_seg / len(dataloaders[phase].dataset)
            epoch_loss_ecr = running_loss_ecr / len(dataloaders[phase].dataset)
            epoch_loss_er = running_loss_er / len(dataloaders[phase].dataset)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.6f}, '
                  'Loss_cls: {:.6f}, '
                  'Loss_cri: {:.6f}, '
                  'Loss_crf: {:.6f}, '
                  'Loss_seg: {:.6f}, '
                  'Loss_ecr: {:.6f}, '
                  'Loss_er: {:.6f}, '.format(phase, epoch_loss, epoch_loss_cls, epoch_loss_cri, epoch_loss_crf, epoch_loss_seg, epoch_loss_ecr, epoch_loss_er))

            if phase == 'val':
                precision_cls = metrics_cls.get_precision()
                print('precision_cls:{:.4f}'.format(precision_cls))
                f1_max = 0.0
                i_max = -1
                for i in range(20):
                    ff = metrics[i].get_f_score()
                    # print(i*0.05, ff)
                    if ff > f1_max:
                        f1_max = ff
                        i_max = i
                precision = metrics[i_max].get_precision()
                recall = metrics[i_max].get_recall()
                f_score = metrics[i_max].get_f_score()
                iou = metrics[i_max].get_fg_iou()
                print('best threshold:{:.2f}'.format(i_max*0.05))
                print('precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}, iou:{:.4f}'.format(precision, recall, f_score, iou))
                if str(type(sc_plt)) == '<class \'torch.optim.lr_scheduler.ReduceLROnPlateau\'>':
                    sc_plt.step(f_score)  # 自适应学习率，根据f1的变化情况调整学习率评价指标
                else:
                    sc_plt.step()

            # Update Scheduler if training loss doesn't change for patience(2) epochs
            if phase == 'train':
                train_loss.append(epoch_loss)
                print('lr:{}'.format(optimizer.param_groups[0]['lr']))

            # deep copy the model and save if F1 is better

            if phase == 'val' and (f_score > best_F1):
                if f_score > best_F1:
                    best_F1 = f_score
                best_checkpoint = '{}/best_model_epoch{:03d}_f_score{:.4f}_iou{:.4f}_cls_precision{:.4f}.pth'.format(path_checkpoints,epoch, f_score, iou, precision_cls)
                torch.save(model, best_checkpoint)
            if phase == 'val':
                torch.save(model, '{}/last_model.pth'.format(path_checkpoints))
                val_acc.append(f_score)
                val_loss.append(epoch_loss)
        print('Best f_score: {:4f}\n'.format(best_F1))

        # acc & loss figure
        # x = np.arange(0, num_epochs, 1)
        x = np.arange(begin_epoch, epoch+1, 1)
        plt.figure()
        plt.plot(x, val_acc, 'r', label='val_pre')
        plt.savefig('{}/val_pre.png'.format(path_figure))
        plt.figure()
        plt.plot(x, train_loss, 'g', label='train_loss')
        plt.savefig('{}/train_loss.png'.format(path_figure))
        plt.figure()
        plt.plot(x, val_loss, 'g', label='val_loss')
        plt.savefig('{}/val_loss.png'.format(path_figure))
        plt.close('all')