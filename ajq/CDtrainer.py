import os
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from utils.metrics import Metrics


def train_model(model, dataloaders, criterion, criterion1, optimizer, scheduler, device, num_epochs, save_path=None, base_lr=1e-3, power=0.9):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_F1 = 0.0
    best_epoch = 0
    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            metrics = Metrics(range(2))
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss_seg = 0.0
            running_loss_edge = 0.0

            for sample in tqdm(dataloaders[phase]):
                image1 = sample['A'].to(device)
                image2 = sample['B'].to(device)
                labels = (sample['label'] > 0).type(torch.LongTensor).to(device)
                labels_edge = (sample['label_edge'] > 0).type(torch.LongTensor).to(device)

                # egrcnn
                # labels_edge_2 = (sample['label_edge'] > 0).type(torch.LongTensor).to(device)
                # labels_edge_1 = torch.ones(
                #     (labels_edge_2.shape[0], 1, labels_edge_2.shape[2], labels_edge_2.shape[3])).type(
                #     torch.LongTensor).to(device)
                # labels_edge_1 = torch.sub(labels_edge_1, labels_edge_2)
                # labels_edge = torch.cat((labels_edge_1, labels_edge_2), dim=1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # pred = model(image1, image2)
                    # loss = criterion(pred, labels)

                    # pred, edge = model(image1, image2)
                    # loss = criterion(pred, labels) + 0.25 * criterion(edge, labels_edge)
                    # ---------------------------------------------------------------------
                    # p2v
                    # pred, pred_v = model(image1, image2)
                    # loss = criterion(pred, labels) + 0.4 * criterion(pred_v, labels)
                    # ---------------------------------------------------------------------
                    # dsamnet
                    # pred, aux_pred_1, aux_pred_2 = model(image1, image2)
                    # loss = criterion(pred, labels) + 0.1 * criterion1(aux_pred_1, labels) + 0.1 * criterion1(aux_pred_2, labels)
                    # pred = pred.cpu().detach().numpy()
                    # pred[pred >= 1.0] = 1.0
                    # pred[pred < 1.0] = 0.0
                    # ---------------------------------------------------------------------
                    # DMINet
                    pred_1, pred_2, pred_1_2, pred_2_2 = model(image1, image2)
                    loss = criterion(pred_1, labels) + criterion(pred_2, labels) + 0.8 * criterion(pred_1_2, labels) + 0.8 * criterion(pred_2_2, labels)
                    pred = pred_1 + pred_2
                    # ---------------------------------------------------------------------
                    #EGRCNN
                    # image1 = image1.unsqueeze(0)
                    # image2 = image2.unsqueeze(0)
                    # image_input = torch.cat((image1, image2), 0)
                    # d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge = model(image_input)
                    # loss_seg_2 = criterion(d2_out, labels)
                    # loss_seg_3 = criterion(d3_out, labels)
                    # loss_seg_4 = criterion(d4_out, labels)
                    # loss_seg_5 = criterion(d5_out, labels)
                    # loss_seg_6 = criterion(d6_out, labels)
                    # loss_edge_2 = criterion1(F.softmax(d2_edge, dim=1), labels_edge.float())  # mse_loss
                    # loss_edge_3 = criterion1(F.softmax(d3_edge, dim=1), labels_edge.float())
                    # loss_edge = 10 * (loss_edge_2 + loss_edge_3)
                    # loss_seg = loss_seg_2 + loss_seg_3 + loss_seg_4 + loss_seg_5 + loss_seg_6
                    # loss = loss_edge + loss_seg
                    # pred = d2_out

                    for mask, output in zip(labels, pred):
                        metrics.add(mask, output)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss_seg += loss.item()
                running_loss_edge += loss.item()

            epoch_loss_seg = running_loss_seg / len(dataloaders[phase].dataset)
            epoch_loss = epoch_loss_seg
            print('{} Loss_seg: {:.6f}'.format(phase, epoch_loss_seg))

            if phase == 'train':
                train_loss.append(epoch_loss)
                precision = metrics.get_precision()
                recall = metrics.get_recall()
                f_score = metrics.get_f_score()
                oa = metrics.get_oa()
                miou = metrics.get_miou()
                print(
                    'precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}, oa:{:.4f}, miou:{:.4f}'.format(precision, recall, f_score, oa, miou))
                train_acc.append(f_score)
                scheduler.step()

            if phase == 'val':
                val_loss.append(epoch_loss)
                precision = metrics.get_precision()
                recall = metrics.get_recall()
                f_score = metrics.get_f_score()
                oa = metrics.get_oa()
                miou = metrics.get_miou()
                print(
                    'precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}, oa:{:.4f}, miou:{:.4f}'.format(precision, recall, f_score, oa, miou))
                val_acc.append(f_score)
                print('lr:{}'.format(optimizer.param_groups[0]['lr']))

                if f_score > best_F1:
                    best_F1 = f_score
                    best_epoch = epoch
                    model_path = save_path
                    model_name = 'best_model_epoch.pth'
                    torch.save(model, os.path.join(model_path, model_name))
                    print('Updata best epoch, best f_score: {:4f}'.format(best_F1))
                if f_score > 0.90:
                    model_path = save_path
                    model_name = str(f_score) + '_model_checkpoints.pth'
                    torch.save(model, os.path.join(model_path, model_name))
        print('Best f_score: {:4f} at best epoch: {}'.format(best_F1, best_epoch + 1))
    return train_loss, val_loss, train_acc, val_acc
