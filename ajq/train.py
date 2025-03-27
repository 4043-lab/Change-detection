
import os

from models.Bitnet import BIT
from models.DMINet import DMINet
from models.FC_EF import Unet
from models.FC_siam_conc import SiamUnet_conc
from models.FC_siam_diff import SiamUnet_diff
# from models.SEHR import SEHR
from models.TDCCNet import TDCC
from models.dsamnet import DSAMNet
from models.egrcnn import EGRCNN
from models.p2v import P2VNet
from models.snunet import SNUNet

from utils import losses

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms

import CDtrainer
import dataset
import warnings
warnings.filterwarnings('ignore')

num_epochs = 100
num_classes = 2
batch_size = 4
img_size = 256
base_lr = 1e-2
power = 0.9

# dataset_name = 'LEVIR-CD_256'
# dataset_name = 'CDD'
# dataset_name = 'BCDD'
dataset_name = 'SYSU-CD'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(3407)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_ckpt = False
model_name = 'test_DMINet_sgd_1e-2_decay05_bs4_SYSU'
save_path = os.path.join('checkpoints' + model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_path = os.path.join(dataset_name, 'train')
val_path = os.path.join(dataset_name, 'val')
test_path = os.path.join(dataset_name, 'test')


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ]),
}

if dataset_name == 'LEVIR-CD_256' or dataset_name == 'LEVIR-CD' or dataset_name == 'BCDD':
    train_dataset = dataset.LEVIR_CD(train_path, data_transforms['train'])
    val_dataset = dataset.LEVIR_CD(test_path, data_transforms['val'])
elif dataset_name == 'CDD':
    train_dataset = dataset.CDD(train_path, data_transforms['train'])
    val_dataset = dataset.CDD(test_path, data_transforms['val'])
elif dataset_name == 'SYSU-CD':
    train_dataset = dataset.SYSU_Dataset(train_path, data_transforms['train'])
    val_dataset = dataset.SYSU_Dataset(test_path, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True, pin_memory=True)
dataloaders_dict = {'train': train_loader, 'val': val_loader}

# model = TDCC()
# model = SEHR()
# model = BIT(3, 2)
# model = SNUNet()
# model = P2VNet(3)
# model = DSAMNet(3, 2)
# model = Unet(6, 2)
# model = SiamUnet_diff(3, 2)
# model = SiamUnet_conc(3, 2)
model = DMINet()
# model = EGRCNN()

if load_ckpt:
    checkpoint_path = os.path.join(save_path, '0.9159388351270755_model_checkpoints.pth')
    model = torch.load(checkpoint_path)

model = model.to(device)

criterion = losses.BCELoss()
# criterion = losses.BCLLoss()
criterion1 = nn.MSELoss()


optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

train_loss, val_loss, train_score, val_score = CDtrainer.train_model(model, dataloaders_dict, criterion, criterion1, optimizer, scheduler, device, num_epochs=num_epochs, save_path=save_path, base_lr=base_lr, power=power)

epoch = np.arange(0, num_epochs, 1)
plt.figure()
plt.plot(epoch, train_loss, 'r', label='train_loss')
plt.plot(epoch,val_loss, 'g', label='val_loss')
plt.savefig(os.path.join(save_path, 'loss.png'))
plt.figure()
plt.plot(epoch, train_score, 'r', label='train_score')
plt.plot(epoch, val_score, 'g', label='val_score')
plt.savefig(os.path.join(save_path, 'score.png'))
