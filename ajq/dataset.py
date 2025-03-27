from torch.utils.data import Dataset
import torchvision
from PIL import Image
import glob

class LEVIR_CD(Dataset):
    def __init__(self, file, transform=None):

        image_path1 = glob.glob(file + '/A' + '/*.png')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(file + '/B' + '/*.png')
        image_path2.sort()
        self.image_path2 = image_path2

        label_path = glob.glob(file + '/label' + '/*.png')
        label_path.sort()
        self.label_path = label_path

        label_edge_path = glob.glob(file + '/edge' + '/*.png')
        label_edge_path.sort()
        self.label_edge_path = label_edge_path

        self.transform = transform

    def __len__(self):
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        label = Image.open(self.label_path[idx])
        # sample = {'A': images1, 'B': images2, 'label': label}
        label_edge = Image.open(self.label_edge_path[idx])
        sample = {'A': images1, 'B': images2, 'label': label, 'label_edge': label_edge}

        if self.transform:
            image1 = sample['A']
            image2 = sample['B']
            mask = sample['label']
            mask_edge = sample['label_edge']

            for t in self.transform.transforms:
                image1 = t(image1)
                image2 = t(image2)

                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        mask = t(mask) * 255.0
                        mask_edge = t(mask_edge) * 255.0
                    else:
                        if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                            if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                mask = t(mask)
                                mask_edge = t(mask_edge)
            # sample = {'A': image1, 'B': image2, 'label': mask}
            sample = {'A': image1, 'B': image2, 'label': mask, 'label_edge': mask_edge}

        return sample


class CDD(Dataset):
    def __init__(self, file, transform=None):

        image_path1 = glob.glob(file + '/A' + '/*.jpg')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(file + '/B' + '/*.jpg')
        image_path2.sort()
        self.image_path2 = image_path2

        label_path = glob.glob(file + '/label' + '/*.jpg')
        label_path.sort()
        self.label_path = label_path

        label_edge_path = glob.glob(file + '/edge' + '/*.jpg')
        label_edge_path.sort()
        self.label_edge_path = label_edge_path

        self.transform = transform

    def __len__(self):
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        label = Image.open(self.label_path[idx])
        # sample = {'A': images1, 'B': images2, 'label': label}
        label_edge = Image.open(self.label_edge_path[idx])
        sample = {'A': images1, 'B': images2, 'label': label, 'label_edge': label_edge}

        if self.transform:
            image1 = sample['A']
            image2 = sample['B']
            mask = sample['label']
            mask_edge = sample['label_edge']

            for t in self.transform.transforms:
                image1 = t(image1)
                image2 = t(image2)

                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        mask = t(mask) * 255.0
                        mask_edge = t(mask_edge) * 255.0
                    else:
                        if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                            if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                mask = t(mask)
                                mask_edge = t(mask_edge)
            # sample = {'A': image1, 'B': image2, 'label': mask}
            sample = {'A': image1, 'B': image2, 'label': mask, 'label_edge': mask_edge}

        return sample


class DSIFN_Dataset(Dataset):
    def __init__(self, file, transform=None):
        super(DSIFN_Dataset, self).__init__()
        image_path1 = glob.glob(file + '/A' + '/*.jpg')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(file + '/B' + '/*.jpg')
        image_path2.sort()
        self.image_path2 = image_path2

        label_path = glob.glob(file + '/label' + '/*.png')
        label_path.sort()
        self.label_path = label_path

        self.transform = transform

    def __len__(self):
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        label = Image.open(self.label_path[idx])
        sample = {'A': images1, 'B': images2, 'label': label}

        if self.transform:
            image1 = sample['A']
            image2 = sample['B']
            mask = sample['label']

            for t in self.transform.transforms:
                image1 = t(image1)
                image2 = t(image2)

                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        mask = t(mask) * 255.0
                    else:
                        if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                            if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                mask = t(mask)
            sample = {'A': image1, 'B': image2, 'label': mask}

        return sample


class SYSU_Dataset(Dataset):
    def __init__(self, file, transform=None):
        super(SYSU_Dataset, self).__init__()
        image_path1 = glob.glob(file + '/A' + '/*.png')
        image_path1.sort()
        self.image_path1 = image_path1

        image_path2 = glob.glob(file + '/B' + '/*.png')
        image_path2.sort()
        self.image_path2 = image_path2

        label_path = glob.glob(file + '/label' + '/*.png')
        label_path.sort()
        self.label_path = label_path

        label_edge_path = glob.glob(file + '/edge' + '/*.png')
        label_edge_path.sort()
        self.label_edge_path = label_edge_path

        self.transform = transform

    def __len__(self):
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        label = Image.open(self.label_path[idx])
        # sample = {'A': images1, 'B': images2, 'label': label}
        label_edge = Image.open(self.label_edge_path[idx])
        sample = {'A': images1, 'B': images2, 'label': label, 'label_edge': label_edge}

        if self.transform:
            image1 = sample['A']
            image2 = sample['B']
            mask = sample['label']
            mask_edge = sample['label_edge']

            for t in self.transform.transforms:
                image1 = t(image1)
                image2 = t(image2)

                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        mask = t(mask) * 255.0
                        mask_edge = t(mask_edge) * 255.0
                    else:
                        if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                            if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                mask = t(mask)
                                mask_edge = t(mask_edge)
            # sample = {'A': image1, 'B': image2, 'label': mask}
            sample = {'A': image1, 'B': image2, 'label': mask, 'label_edge': mask_edge}

        return sample

