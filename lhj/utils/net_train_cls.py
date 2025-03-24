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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def train_model(model, dataloaders, criterion, criterion_weight, optimizer, sc_plt, num_epochs=100, model_name='',
                model_info='', model_type='', temperature=False, multi_loss=False, dataset_name='', begin_epoch=0, best_pre=0.0):
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

    best_pre = best_pre
    for epoch in range(begin_epoch, num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        metrics = Metrics_cls()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for sample in tqdm(dataloaders[phase]):
            # for sample in dataloaders[phase]:
                t1 = sample['reference'].unsqueeze(0).cuda() #image1
                t2 = sample['test'].unsqueeze(0).cuda() #image2
                t = torch.cat([t1, t2], dim=0)

                labels = (sample['label'] > 0).squeeze(1).type(torch.LongTensor).cuda()
                # print(labels)
                # print(labels.shape)
                labels = torch.sum(labels, dim=(1, 2))
                labels[labels>0] = 1  # [B]
                # print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  # train 计算梯度，val的时候不计算
                    _,_,_,_,out_cls2,out_cls3,out_cls4,out_cls5 = model(t)  # [B, 2]

                    loss = criterion(out_cls2, labels)
                    loss += criterion(out_cls3, labels)
                    loss += criterion(out_cls4, labels)
                    loss += criterion(out_cls5, labels)
                    # print(loss)

                    # Calculate metric during evaluation
                    if phase == 'val':
                        metrics.add(labels, out_cls5)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * t1.size(1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss_seg: {:.6f}'.format(phase, epoch_loss))  # edge

            if phase == 'val':
                precision = metrics.get_precision()
                print('precision:{:.4f}'.format(precision))
                if str(type(sc_plt)) == '<class \'torch.optim.lr_scheduler.ReduceLROnPlateau\'>':
                    sc_plt.step(precision)  # 自适应学习率，根据f1的变化情况调整学习率评价指标
                else:
                    sc_plt.step()

            # Update Scheduler if training loss doesn't change for patience(2) epochs
            if phase == 'train':
                train_loss.append(epoch_loss)
                print('lr:{}'.format(optimizer.param_groups[0]['lr']))

            # deep copy the model and save if F1 is better

            if phase == 'val' and (precision > best_pre):
                if precision > best_pre:
                    best_pre = precision
                best_checkpoint = '{}/best_model_epoch{:03d}_precision{:.4f}.pth'.format(path_checkpoints,epoch, precision)
                torch.save(model, best_checkpoint)
            if phase == 'val':
                torch.save(model, '{}/last_model.pth'.format(path_checkpoints))
                val_acc.append(precision)
                val_loss.append(epoch_loss)
        print('Best precision: {:4f}\n'.format(best_pre))

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