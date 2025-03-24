"# -- coding: UTF-8 --"
from torch.utils.data import Dataset
import numpy as np
import torchvision
from utils import helper_augmentations
from PIL import Image
import glob
import torch


class ChangeDatasetNumpy(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, files, transform=None, t1_name='', t2_name='', label_name='', edge_name=''):

        image_path1 = glob.glob(files + '/' + t1_name + '/*')
        image_path1.sort()
        self.image_path1 = image_path1

        canny_path1 = glob.glob(files + '/' + t1_name + '_canny/*')
        canny_path1.sort()
        self.canny_path1 = canny_path1

        image_path2 = glob.glob(files + '/' + t2_name + '/*')
        image_path2.sort()
        self.image_path2 = image_path2

        canny_path2 = glob.glob(files + '/' + t2_name + '_canny/*')
        canny_path2.sort()
        self.canny_path2 = canny_path2

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
        cannys1 = Image.open(self.canny_path1[idx]).convert('L')
        cannys2 = Image.open(self.canny_path2[idx]).convert('L')
        mask = Image.open(self.target[idx]).convert('L')

        sample = {'reference': images1, 'test': images2, 'label': mask}
        # Handle Augmentations
        if self.transform:
            trf_reference = sample['reference']
            trf_test = sample['test']
            trf_label = sample['label']
            trf_canny1 = cannys1
            trf_canny2 = cannys2
            # Don't do Normalize on label, all the other transformations apply...
            for t in self.transform.transforms:
                if (isinstance(t, helper_augmentations.SwapReferenceTest)) or (
                isinstance(t, helper_augmentations.JitterGamma)):
                    trf_reference, trf_test = t(sample)
                else:
                    # All other type of augmentations
                    trf_reference = t(trf_reference)
                    trf_test = t(trf_test)

                # Don't Normalize or Swap
                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    # ToTensor divide every result by 255
                    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#to_tensor
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        trf_label = t(trf_label) * 255.0
                        trf_canny1 = t(trf_canny1) * 255.0
                        trf_canny2 = t(trf_canny2) * 255.0
                    else:
                        if not isinstance(t, helper_augmentations.SwapReferenceTest):
                            if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                                if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                    if not isinstance(t, helper_augmentations.JitterGamma):
                                        trf_label = t(trf_label)
                                        trf_canny1 = t(trf_canny1)
                                        trf_canny2 = t(trf_canny2)

            # sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label, 'label_edge': trf_label_edge}
            # print(trf_canny1.shape)
            # print(trf_reference.shape)
            sample = {'reference': torch.cat([trf_reference, trf_canny1], dim=0), 'test': torch.cat([trf_test, trf_canny2], dim=0), 'label': trf_label}

        return sample