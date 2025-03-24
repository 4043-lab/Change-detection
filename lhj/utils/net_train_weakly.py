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
from utils.metrics import Metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def train_model(model, dataloaders, criterion, criterion_weight, optimizer, sc_plt, num_epochs=100, model_name='',
                model_info='', model_type='', dataset_name='', begin_epoch=0, best_f1=0.0):
    model_root = 'result/{}/{}/{}'.format(model_type, model_name, dataset_name)
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

    best_F1 = best_f1
    # best_F1 = 0.0
    # best_iou = 0.0
    for epoch in range(begin_epoch, num_epochs):
        # print('lr={}'.format(optimizer.))
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        metrics = Metrics()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_consistency = 0.0
            running_loss_saliency = 0.0

            # Iterate over data.
            for sample in tqdm(dataloaders[phase]):
            # for sample in dataloaders[phase]:
                img1 = sample['reference'].cuda() #image1
                img2 = sample['test'].cuda() #image2
                # labels = (sample['label'] > 0).squeeze(1).type(torch.LongTensor).to(device)
                labels = (sample['label'] > 0).type(torch.FloatTensor).cuda()
                sp1 = sample['label_sp1'].cuda()
                sp2 = sample['label_sp2'].cuda()
                num1 = sample['num_sp1'].cuda()
                num2 = sample['num_sp2'].cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  # train 计算梯度，val的时候不计算
                    # Get model outputs and calculate loss

                    image_out = model(img1, img2)

                    # for i in image_out:
                    #     print(i.shape)
                    # print(image_out.shape)
                    # print(labels.shape)

                    loss_c1 = criterion_weight[0] * criterion[0](img1, image_out[0], sp1, num1)
                    loss_c2 = criterion_weight[1] * criterion[1](img2, image_out[1], sp2, num2)
                    loss_s = criterion_weight[2] * criterion[2](image_out[2], image_out[3])
                    loss = loss_c1 + loss_c2 + loss_s
                    # print(loss_c1, loss_c2, loss_s)
                    # loss = loss_c1 + loss_c2
                    # print(loss)

                    # Calculate metric during evaluation
                    if phase == 'val':
                        if isinstance(image_out, (list, tuple)):
                            # out = image_out[0] + image_out[1]
                            for mask, output in zip(labels, image_out[-1]):
                                metrics.add(mask, output)
                        else:
                            for mask, output in zip(labels, image_out):
                                metrics.add(mask, output)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # running_loss += loss.item() * reference_img.size(1)
                running_loss += loss.item()
                running_loss_consistency += (loss_c1.item() + loss_c2.item())
                running_loss_saliency += loss_s.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_loss_consistency = running_loss_consistency / len(dataloaders[phase].dataset)
            epoch_loss_saliency = running_loss_saliency / len(dataloaders[phase].dataset)

            print('{} Loss_seg: {:.6f}'.format(phase, epoch_loss))
            print('{} Loss_consistency: {:.6f}'.format(phase, epoch_loss_consistency))
            print('{} Loss_saliency: {:.6f}'.format(phase, epoch_loss_saliency))

            if phase == 'val':
                precision = metrics.get_precision()
                recall = metrics.get_recall()
                f_score = metrics.get_f_score()
                iou = metrics.get_fg_iou()
                oa = metrics.get_oa()
                print(
                    'precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}, oa:{:.4f}, iou:{:.4f}'.format(precision, recall, f_score, oa, iou))
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
                best_checkpoint = '{}/best_model_epoch{:03d}_f_score{:.4f}_iou{:.4f}.pth'.format(path_checkpoints,epoch, f_score, iou)
                torch.save(model, best_checkpoint)
            # if phase == 'val':
                # if (f_score > best_F1) or (iou > best_iou):
                #     best_checkpoint = 'result/checkpoints/best_model_epoch{}_f_score{:.4f}_iou{:.4f}.pth'.format(epoch, f_score, iou)
                #     torch.save(model, best_checkpoint)
                # if f_score > best_F1:
                #     best_F1 = f_score
                # if iou > best_iou:
                #     best_iou = iou
            if phase == 'val':
                torch.save(model, '{}/last_model.pth'.format(path_checkpoints))
                val_acc.append(f_score)
                val_loss.append(epoch_loss)
                # val_acc.append(iou)
        print('Best f_score: {:4f}\n'.format(best_F1))

        # acc & loss figure
        # x = np.arange(0, num_epochs, 1)
        x = np.arange(begin_epoch, epoch+1, 1)
        plt.figure()
        plt.plot(x, val_acc, 'r', label='val_f1')
        plt.savefig('{}/val_f1.png'.format(path_figure))
        plt.figure()
        plt.plot(x, train_loss, 'g', label='train_loss')
        plt.savefig('{}/train_loss.png'.format(path_figure))
        plt.figure()
        plt.plot(x, val_loss, 'g', label='val_loss')
        plt.savefig('{}/val_loss.png'.format(path_figure))
        plt.close('all')
        # return val_acc, train_loss