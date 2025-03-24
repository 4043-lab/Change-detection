"# -- coding: UTF-8 --"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
num_workers = 0
batch_size = 4
base_lr = 1e-3
seed = 20

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import random
from utils import loss,net_train,net_train_no_edge,net_train_proto,loss_proto
from utils import change_dataset_np,change_dataset_np_no_edge
# from model import model_test,FC_EF,FC_siam_diff,FC_siam_conc,STANet,SNUNet,FCCDN,ConvNext,BIT,ChangeFormer
# from model.EGFDFN import dual_resnet_1se_dfm,final_SE_dual_dfm_CBAM,final_SE_dual_CBAM_dfm,final_SE_dual_CBAM_dfm_2,final_SE_dual_CBAM_dfm_4,EGFDFN_condconv_1
# from model.Changer import ChangerVanilla
# from model.idea import T1_add_T2, EGFDFN_add, CondConv, CondConv_3, CondConv_4, CondConv_5, CondConv_6, CondConv_7, CondConv_8, CondConv_9, DynamicConv, DynamicConv_2, DynamicConv_3, DynamicConv_pro, ODConv_1, ODConv_2, ODConv_3
# from model.baseline import resnet_bottle,resnet_bottle_3,resnet_bottle_4,resnet_bottle_5,resnet_bottle_6,resnet50_1
# from models import EGRCNN
# from model.Attention_test import resnet50_SE_1,resnet50_SE_2,resnet50_CBAM_1,resnet50_CBAM_2,resnet50_ECA_1,resnet50_CrissCross_1
# from model import P2V
# from models import snunet
# from model.idea2 import resnet50_cond_1, resnet50_cond_2, resnet50_cond_3, resnet50_cond_CEEF_1, resnet50_cond_SE_1, resnet_new_cond_1, resnet_new_cond_2, resnet_new_cond_3, resnet_new_cond_RViT_1, RViT_only_1, RViT_only_2, RViT_only_3, RViT_only_4, RViT_only_5, resnet50_cond_RViT_1
# from model.idea2 import resnet_new2, resnet_new2_cond, resnet_new2_cond_RViT_1, resnet_new2_cond_RViT_2, resnet_new2_cond_edge
# from model.idea2 import RViT_new_only_1, ConvLSTM_only_1, ConvLSTM_new_only_1, ConvLSTM_new2_only_1, ConvLSTM_new2_only_2, ConvLSTM_new2_only_3, ConvLSTM_new2_only_4, ConvLSTM_new2_only_5, ConvLSTM_new2_only_6
# from model.idea2 import final_cond_ConvLSTM_1, final_cond_ConvLSTM_2, final_cond_ConvLSTM_3, final_cond_ConvLSTM_4, final_cond_ConvLSTM_5, final_cond_ConvLSTM_6
# from model.idea2 import final_cond_ConvLSTM_cross_1, final_cond_ConvLSTM_cross_2, final_cond_ConvLSTM_AW_2, final_cond_ConvLSTM_AW_3, final_cond_ConvLSTM_AW_4
# from model.idea2 import final_cond_ConvLSTM_AW_new_1
# from model.idea2 import resnet_cond_proto_1
# from model.idea2 import resnet50_cond_up_1, resnet50_cond_up_2, resnet50_cond_up_4, resnet50_cond_SE_up_1, resnet50_cond_ECA_up_1
# from model.idea2 import resnet50_cond_ECA_1
# from model.idea2.ablation import resnet50_cond_k4,resnet50_cond_k0,resnet50_cond_k2,resnet50_cond_k8,resnet50_cond_k128
# from model.idea2.ablation import resnet50_cond_up_bicubic, resnet50_cond_up_nearest
# from model.idea2.ablation import resnet50
# from model.idea3 import resnet50, resnet50_head128, resnet50_maxpool, resnet50_dsc, resnet50_dsc_2, resnet50_dsc_3, resnet50_simple, resnet50_simple_dsc, resnet50_simple_dsc_2, resnet50_dsc_cond
from model.EGFDFN import resnet_simple_dsc, resnet_simple_dsc_CBAM_dfm, resnet_simple_dsc_CBAM_dfm_2, resnet_simple_dsc_CBAM
from model.idea4_new import resnet50_fully
print("PID:",os.getpid())
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Hyperparameters
num_epochs = 200
# num_classes = 2
img_size = 256
temperature = False
edge = False
pin_memory_non_blocking = False

dataset_name = 'LEVIR-CD-256'
# dataset_name = 'LEVIR-CD-256-1'
# dataset_name = 'LEVIR-CD-256-overlap'
# dataset_name = 'CDD'
# dataset_name = 'BCDD'
# dataset_name = 'SYSU-CD'
# dataset_name = 'CD_GZ'
# dataset_name = 'TEST'
t1_name = ''
t2_name = ''
label_name = ''
edge_name = 'edge'

if dataset_name == 'CDD':
    train_pickle_file = 'dataset/ChangeDetectionDataset/Real/subset/train'  # Path to training set
    val_pickle_file = 'dataset/ChangeDetectionDataset/Real/subset/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'OUT'
elif dataset_name == 'LEVIR-CD-256':
    train_pickle_file = 'dataset/LEVIR-CD-256/train'  # Path to training set
    val_pickle_file = 'dataset/LEVIR-CD-256/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'label'
elif dataset_name == 'LEVIR-CD-256-1':
    train_pickle_file = 'dataset/LEVIR-CD-256-1/train'  # Path to training set
    val_pickle_file = 'dataset/LEVIR-CD-256-1/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'label'
elif dataset_name == 'LEVIR-CD-256-overlap':
    train_pickle_file = 'dataset/LEVIR-CD-256-overlap/train'  # Path to training set
    val_pickle_file = 'dataset/LEVIR-CD-256/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'label'
elif dataset_name == 'BCDD':
    train_pickle_file = 'dataset/BCDD/train'  # Path to training set
    val_pickle_file = 'dataset/BCDD/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'label'
elif dataset_name == 'SYSU-CD':
    train_pickle_file = 'dataset/SYSU-CD/train'  # Path to training set
    val_pickle_file = 'dataset/SYSU-CD/test'  # Path to validation set
    t1_name = 'time1'
    t2_name = 'time2'
    label_name = 'label'
elif dataset_name == 'CD_GZ':
    train_pickle_file = 'dataset/CD_GZ/train'  # Path to training set
    val_pickle_file = 'dataset/CD_GZ/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'label'
elif dataset_name == 'TEST':
    train_pickle_file = 'dataset/dataset_test/train'  # Path to training set
    val_pickle_file = 'dataset/dataset_test/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
    label_name = 'label'
else:
    raise NotImplementedError()

# train_pickle_file = 'dataset/dataset_test/train' #Path to training set
# val_pickle_file = 'dataset/dataset_test/val' #Path to validation set


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
num_gpu = torch.cuda.device_count()
print('Number of GPUs Available:', num_gpu)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ]),
}

# Create training and validation datasets

if edge:
    train_dataset = change_dataset_np.ChangeDatasetNumpy(train_pickle_file, data_transforms['train'],t1_name=t1_name,t2_name=t2_name,label_name=label_name,edge_name=edge_name)
    val_dataset = change_dataset_np.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'],t1_name=t1_name,t2_name=t2_name,label_name=label_name,edge_name=edge_name)
else:
    train_dataset = change_dataset_np_no_edge.ChangeDatasetNumpy(train_pickle_file, data_transforms['train'],t1_name=t1_name,t2_name=t2_name,label_name=label_name)
    val_dataset = change_dataset_np_no_edge.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'],t1_name=t1_name,t2_name=t2_name,label_name=label_name)
image_datasets = {'train': train_dataset, 'val': val_dataset}
# Create training and validation dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_non_blocking)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_non_blocking)
dataloaders_dict = {'train': train_loader, 'val': val_loader}

# Initialize Model
model_type = 'idea4_new'
model_name = 'resnet50_fully'
model_info = '''resnet50_fully
image size: 256
optim: Adam
loss: BCE
base_lr: 1e-3
lr_schedule: ReduceLROnPlateau
head: 256
init: kaiming
rand_seed: 20
edge: False
'''
# model_name = 'TEST'
# model_info = '''TEST
# '''

# net = final_SE_dual_CBAM_dfm_2.EGFDFN(3, 2, 256)  #网络
# net = model_test.Changer(in_ch=3,size=img_size,pretrained=True)
# net = FC_EF.Unet(6,2)
# net = FC_siam_diff.SiamUnet_diff(3,2)
# net = FC_siam_conc.SiamUnet_conc(3,2)
# net = STANet.STANet(3,att_type='PAM')
# net = snunet.SNUNet(in_ch=3,out_ch=2)
# net = FCCDN.FCCDN(3,16,True)
# net = ChangerVanilla.Changer(in_ch=3,size=img_size,pretrained=True)
# net = T1_add_T2.UNet(img_ch=3, output_ch=2, size=img_size)
# net = EGFDFN_add.EGFDFN(img_ch=32, output_ch=2, patch_size=img_size)
# net = ConvNext.ConvNext_CD(img_ch=3, output_ch=2, patch_size=img_size)
# net = resnet_bottle.Resnet_bottle()
# net = resnet_bottle_3.Resnet_bottle()
# net = resnet_bottle_4.Resnet_bottle()
# net = resnet_bottle_5.Resnet_bottle()
# net = resnet_bottle_6.Resnet_bottle()
# net = resnet50_1.Resnet_bottle()
# net = CondConv.Resnet_bottle()
# net = CondConv_3.Resnet_bottle()
# net = CondConv_4.Resnet_bottle()
# net = CondConv_5.Resnet_bottle()
# net = CondConv_6.Resnet_bottle()
# net = CondConv_7.Resnet_bottle()
# net = CondConv_8.Resnet_bottle()
# net = CondConv_9.Resnet_bottle()
# net = DynamicConv.Resnet_bottle()
# net = DynamicConv_2.Resnet_bottle()
# net = DynamicConv_3.Resnet_bottle()
# net = DynamicConv_pro.Resnet_bottle()
# net = ODConv_1.Resnet_bottle()
# net = ODConv_2.Resnet_bottle()
# net = ODConv_3.Resnet_bottle()
# net = EGFDFN_condconv_1.Resnet_bottle()
# net = BIT.BASE_Transformer(3,2,'learned')
# net = EGRCNN.EGRCNN_Net()
# net = ChangeFormer.ChangeFormerV6()
# net = resnet50_SE_1.Resnet_bottle()
# net = resnet50_SE_2.Resnet_bottle()
# net = resnet50_CBAM_1.Resnet_bottle()
# net = resnet50_CBAM_2.Resnet_bottle()
# net = resnet50_ECA_1.Resnet_bottle()
# net = resnet50_CrissCross_1.Resnet_bottle()
# net = resnet50_cond_1.Network()
# net = resnet50_cond_2.Network()
# net = resnet50_cond_3.Network()
# net = resnet50_cond_CEEF_1.Network()
# net = resnet50_cond_SE_1.Network()
# net = resnet_new_cond_1.Network()
# net = resnet_new_cond_2.Network()
# net = resnet_new_cond_3.Network()
# net = resnet_new_cond_RViT_1.Network()
# net = RViT_only_1.Network()
# net = RViT_only_2.Network()
# net = RViT_only_3.Network()
# net = RViT_only_4.Network()
# net = RViT_only_5.Network()
# net = RViT_new_only_1.Network()
# net = resnet50_cond_RViT_1.Network()
# net = P2V.P2VNet(in_ch=3)
# net = resnet_new2.Network()
# net = resnet_new2_cond.Network()
# net = resnet_new2_cond_edge.Network()
# net = resnet_new2_cond_RViT_1.Network()
# net = resnet_new2_cond_RViT_2.Network()
# net = resnet_cond_proto_1.Network()
# net = ConvLSTM_only_1.Network(seq_len=3,num_layers=2)
# net = ConvLSTM_new_only_1.Network(seq_len=4,num_layers=2)
# net = ConvLSTM_new2_only_1.Network(seq_len=4,num_layers=2)
# net = ConvLSTM_new2_only_2.Network(seq_len=4,num_layers=2)
# net = ConvLSTM_new2_only_3.Network(seq_len=5,num_layers=2)
# net = ConvLSTM_new2_only_4.Network(seq_len=4,num_layers=2)
# net = ConvLSTM_new2_only_5.Network(seq_len=4,num_layers=2)
# net = ConvLSTM_new2_only_6.Network(seq_len=2,num_layers=2)
# net = final_cond_ConvLSTM_1.Network()
# net = final_cond_ConvLSTM_2.Network()
# net = final_cond_ConvLSTM_3.Network()
# net = final_cond_ConvLSTM_4.Network()
# net = final_cond_ConvLSTM_5.Network()
# net = final_cond_ConvLSTM_6.Network()
# net = final_cond_ConvLSTM_cross_1.Network()
# net = final_cond_ConvLSTM_cross_2.Network()
# net = final_cond_ConvLSTM_AW_2.Network()
# net = final_cond_ConvLSTM_AW_3.Network()
# net = final_cond_ConvLSTM_AW_4.Network()
# net = final_cond_ConvLSTM_AW_new_1.Network()
# net = resnet50_cond_up_1.Network()
# net = resnet50_cond_up_2.Network()
# net = resnet50_cond_up_4.Network()
# net = resnet50_cond_ECA_1.Network()
# net = resnet50_cond_SE_up_1.Network()
# net = resnet50_cond_ECA_up_1.Network()
# net = resnet50_cond_k0.Network()
# net = resnet50_cond_k2.Network()
# net = resnet50_cond_k4.Network()
# net = resnet50_cond_k128.Network()
# net = resnet50_cond_k8.Network()
# net = resnet50_cond_up_nearest.Network()
# net = resnet50_cond_up_bicubic.Network()
# net = resnet50.Network()
# net = resnet50_head128.Network()
# net = resnet50_maxpool.Network()
# net = resnet50_dsc.Network()
# net = resnet50_dsc_2.Network()
# net = resnet50_dsc_3.Network()
# net = resnet50_simple.Network()
# net = resnet50_simple_dsc.Network()
# net = resnet50_simple_dsc_2.Network()
# net = resnet50_dsc_cond.Network()
# net = resnet_simple_dsc.Network()
# net = resnet_simple_dsc_CBAM.Network()
# net = resnet_simple_dsc_CBAM_dfm.Network()
net = resnet_simple_dsc_CBAM_dfm_2.Network()
# net = resnet50_fully.Network()

begin_epoch = 0
best_f1 = 0.0
pretained = False
if pretained:
    model_load = "result/idea2/ablation/resnet50_cond_k128/BCDD/checkpoints/last_model.pth"
    net = torch.load(model_load)
    begin_epoch = 300
    best_f1 = 0.933326
# net = net.cuda()
net = net.to(device, non_blocking=pin_memory_non_blocking)

# device_id = [0, 1]
# net = net.to(device, non_blocking=pin_memory_non_blocking)
# net = DataParallel(net, device_ids=device_id, output_device=device)

criterion1 = nn.MSELoss() # edge loss

multi_loss = False
criterion = loss.BCELoss()
# criterion = loss.BCELoss_2()
criterion_weight = 1.0

# multi_loss = True
# # criterion = [loss.BCELoss(), loss_proto.PPC(), loss_proto.PPD()]
# # criterion = [loss.L1Loss_edge(), loss.BCELoss()]
# # criterion = [loss.BCELoss(), loss.BCELoss()]
# criterion = [loss.BCELoss()] * 9
# criterion_weight = [1.0] * 9

optimizer = optim.Adam(net.parameters(), lr=base_lr)
# optimizer = optim.SGD(net.parameters(), lr=base_lr)
# optimizer = optim.Adam(filter(lambda p: p.requires _grad, net.parameters()), lr=base_lr)
# optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.05)
sc_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True, min_lr=1e-4, factor=0.1)
# sc_plt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-4)
# sc_plt = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

# train
print('Train model:'
      '', model_name)
print('Train dataset:', dataset_name)
print('batch_size:', batch_size)
print('base_lr:', base_lr)
print('multi_loss:', multi_loss)
print('optimizer:', optimizer.__class__)
print('random seed:', seed)
if edge:
    net_train.train_model(net, dataloaders_dict, criterion, criterion1, optimizer, sc_plt, num_epochs=num_epochs, model_name=model_name, model_info=model_info, model_type=model_type)
else:
    net_train_no_edge.train_model(net, dataloaders_dict, criterion, criterion_weight, optimizer, sc_plt, num_epochs=num_epochs, model_name=model_name, model_info=model_info, model_type=model_type, temperature=temperature, multi_loss=multi_loss, dataset_name=dataset_name, begin_epoch=begin_epoch, best_f1=best_f1)
    # net_train_proto.train_model(net, dataloaders_dict, criterion, optimizer, sc_plt, num_epochs=num_epochs, model_name=model_name, model_info=model_info, model_type=model_type, temperature=temperature, multi_loss=multi_loss, dataset_name=dataset_name, begin_epoch=begin_epoch)