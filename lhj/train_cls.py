"# -- coding: UTF-8 --"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
num_workers = 0
batch_size = 4
base_lr = 1e-4

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import random
from utils import loss, net_train_cls, net_train_cls_cam, net_train_cls_cam_argmax
from utils import change_dataset_np_cls
from model.idea4 import resnet50_cls, resnet50_cls_2, resnet50_cls_dsc, resnet50_cls_cond, resnet50_true, resnet50_cls_mjs, resnet50_cls_mjs_icr_1, resnet50_cls_mjs_icr_2, resnet50_cls_mjs_icr_3
from model.idea4 import resnet50_true_icr_feature_1, resnet50_true_icr_feature_2, resnet50_true_icr_feature_3, resnet50_true_icr_feature_2_dualcls, resnet50_true_icr_feature_2_dualcls_bottle, resnet50_true_icr_feature_2_dualcls_bottle_largecam, resnet50_true_icr_feature_2_dualcls_reverse
from model.idea4 import resnet50_true_dual
from model.idea4_new import resnet50_icr_largecam, resnet50_icr_largecam_2, resnet50_icr_largecam_affine, resnet50_cam, SEAM, MJS_ICR_Net
print("PID:",os.getpid())
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Hyperparameters
num_epochs = 200
img_size = 256
# num_classes = 2
temperature = False
edge = False
dataset_name = 'LEVIR-CD-256'
# dataset_name = 'LEVIR-CD-256-1'
# dataset_name = 'CDD'
# dataset_name = 'BCDD'
# dataset_name = 'SYSU-CD'
# dataset_name = 'EGY_BCD'
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
    # train_pickle_file = 'dataset/LEVIR-CD-256/val'  # Path to training set
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
elif dataset_name == 'EGY_BCD':
    train_pickle_file = 'dataset/EGY_BCD/train'  # Path to training set
    val_pickle_file = 'dataset/EGY_BCD/test'  # Path to validation set
    t1_name = 'A'
    t2_name = 'B'
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
# setup_seed(3407)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
num_gpu = torch.cuda.device_count()
batch_size *= num_gpu
base_lr *= num_gpu
print('Number of GPUs Available:', num_gpu)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomChoice([
        #     transforms.RandomRotation((0,0)),
        #     transforms.RandomRotation((90,90)),
        #     transforms.RandomRotation((180,180)),
        #     transforms.RandomRotation((270,270))
        # ]),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]),
}

# Create training and validation datasets

train_dataset = change_dataset_np_cls.ChangeDatasetNumpy(train_pickle_file, data_transforms['train'],t1_name=t1_name,t2_name=t2_name,label_name=label_name)
val_dataset = change_dataset_np_cls.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'],t1_name=t1_name,t2_name=t2_name,label_name=label_name)
image_datasets = {'train': train_dataset, 'val': val_dataset}
# Create training and validation dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
dataloaders_dict = {'train': train_loader, 'val': val_loader}

# Initialize Model
model_type = 'idea4_new'
model_name = 'resnet50_cam_10percent'
model_info = '''resnet50_cam_10percent
image size: 256
batch: 4
optim: Adam
lr_schedule: ReduceLROnPlateau
expansion = 4
res_block num = 3
res_block downsample: 1*1 stride=2
init: kaiming
stem: 7*7
encoder num: 3:4:6:3
rand_seed: 20
head_channel: 2048/1024/512->256->2
label_threshold: 10%
'''

# net = resnet50_cls.Network()
# net = resnet50_cls_dsc.Network()
# net = resnet50_cls_cond.Network()
# net = resnet50_true.Network()
# net = resnet50_cls_2.Network()
# net = resnet50_cls_mjs.Network()
# net = resnet50_cls_mjs_icr_1.Network()
# net = resnet50_cls_mjs_icr_2.Network()
# net = resnet50_cls_mjs_icr_3.Network()
# net = resnet50_true_icr_feature_1.Network()
# net = resnet50_true_icr_feature_2.Network()
# net = resnet50_true_icr_feature_2_dualcls.Network()
# net = resnet50_true_icr_feature_2_dualcls_bottle.Network()
# net = resnet50_true_icr_feature_2_dualcls_bottle_largecam.Network()
# net = resnet50_true_icr_feature_2_dualcls_reverse.Network()
# net = resnet50_true_icr_feature_3.Network()
# net = resnet50_true_dual.Network()
net = resnet50_cam.Network()
# net = resnet50_icr_largecam.Network()
# net = resnet50_icr_largecam_2.Network()
# net = resnet50_icr_largecam_affine.Network(affine_type='flip')
# net = resnet50_icr_largecam_affine.Network(affine_type='rotation')
# net = SEAM.Network()
# net = MJS_ICR_Net.Network()

begin_epoch = 0
best_pre = 0.0
best_F1 = 0.0
pretained = False
if pretained:
    model_load = "result/idea4_new/resnet50_icr_largecam/BCDD/checkpoints/best_model_epoch006_f_score0.7790_iou0.6379_cls_precision0.9725.pth"
    net = torch.load(model_load)
    begin_epoch = 7
    best_pre = 0.9725
    best_F1 = 0.7790
net = net.cuda()

criterion = [nn.L1Loss(), loss.CELoss(), loss.BCELoss(), nn.L1Loss(reduction='sum')]
# criterion = [nn.L1Loss(), loss.CELoss(), loss.BCELoss(), nn.MSELoss(reduction='sum')]
criterion_weight = [1.0, 1.0, 1.0, 1.0]
cam = True
# cam_argmax = True
cam_argmax = False
# change_threshold = 0.05
change_threshold = 0.10

optimizer = optim.Adam(net.parameters(), lr=base_lr)
sc_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True, min_lr=1e-4, factor=0.1)

# train
print('Train model:' + model_name)
print('Train dataset:' + dataset_name)
print('batch_size:', batch_size)
print('base_lr:', base_lr)
print('generate_cam:', cam)
print('change_threshold:', change_threshold)
if cam:
    if cam_argmax:
        net_train_cls_cam_argmax.train_model(net, dataloaders_dict, criterion, criterion_weight, optimizer, sc_plt,
                                  num_epochs=num_epochs, model_name=model_name, model_info=model_info,
                                  model_type=model_type, dataset_name=dataset_name, begin_epoch=begin_epoch,
                                  best_F1=best_F1, out_cam=cam, change_threshold=change_threshold)
    else:
        net_train_cls_cam.train_model(net, dataloaders_dict, criterion, criterion_weight, optimizer, sc_plt,
                                  num_epochs=num_epochs, model_name=model_name, model_info=model_info,
                                  model_type=model_type, dataset_name=dataset_name, begin_epoch=begin_epoch,
                                  best_F1=best_F1, out_cam=cam, change_threshold=change_threshold)
else:
    net_train_cls.train_model(net, dataloaders_dict, criterion, criterion_weight, optimizer, sc_plt,
                              num_epochs=num_epochs, model_name=model_name, model_info=model_info,
                              model_type=model_type, dataset_name=dataset_name, begin_epoch=begin_epoch,
                              best_pre=best_pre)