"# -- coding: UTF-8 --"
from torch.utils.data import Dataset
import numpy as np
import torchvision
from utils import helper_augmentations
from PIL import Image
import glob
import torch
import cv2

def get_superpixel_label(img):
    superpixel = cv2.ximgproc.createSuperpixelSEEDS(image_width=256, image_height=256, image_channels=3, num_levels=10,
                                              num_superpixels=256)
    superpixel.iterate(img, 10)  # 迭代次数，越大效果越好
    label_superpixel = superpixel.getLabels()                       #获取超像素标签
    number_superpixel = superpixel.getNumberOfSuperpixels()         #获取超像素数目
    return label_superpixel, number_superpixel

class ChangeDatasetNumpy(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, files, transform=None, t1_name='', t2_name='', label_name='', edge_name=''):

        image_path1 = glob.glob(files + '/' + t1_name + '/*')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(files + '/' + t2_name + '/*')
        image_path2.sort()
        self.image_path2 = image_path2

        target = glob.glob(files + '/' + label_name + '/*')
        target.sort()
        self.target = target

        self.transform = transform

    def __len__(self):
        # return len(self.data_dict)
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        mask = Image.open(self.target[idx]).convert('L')

        img_sp1 = cv2.cvtColor(np.array(images1), cv2.COLOR_RGB2BGR)
        img_sp2 = cv2.cvtColor(np.array(images2), cv2.COLOR_RGB2BGR)

        sp1, num1 = get_superpixel_label(img_sp1)
        sp2, num2 = get_superpixel_label(img_sp2)
        sp1 = Image.fromarray(sp1)
        sp2 = Image.fromarray(sp2)
        # sp1.show()
        # sp2.show()

        sample = {'reference': images1, 'test': images2, 'label': mask, 'label_sp1': sp1, 'label_sp2': sp2, 'num_sp1': num1, 'num_sp2': num2}
        # Handle Augmentations
        if self.transform:
            trf_reference = sample['reference']
            trf_test = sample['test']
            trf_label = sample['label']
            trf_sp1 = sample['label_sp1']
            trf_sp2 = sample['label_sp2']
            # Don't do Normalize on label, all the other transformations apply...
            for t in self.transform.transforms:
                if (isinstance(t, helper_augmentations.SwapReferenceTest)) or (
                isinstance(t, helper_augmentations.JitterGamma)):
                    trf_reference, trf_test = t(sample)
                else:
                    # All other type of augmentations
                    trf_reference = t(trf_reference)
                    trf_test = t(trf_test)
                    trf_sp1 = t(trf_sp1)
                    trf_sp2 = t(trf_sp2)

                # Don't Normalize or Swap
                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    # ToTensor divide every result by 255
                    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#to_tensor
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        trf_label = t(trf_label) * 255.0
                    else:
                        if not isinstance(t, helper_augmentations.SwapReferenceTest):
                            if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                                if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                    if not isinstance(t, helper_augmentations.JitterGamma):
                                        trf_label = t(trf_label)

            sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label, 'label_sp1': trf_sp1, 'label_sp2': trf_sp2, 'num_sp1': num1, 'num_sp2': num2}

        return sample