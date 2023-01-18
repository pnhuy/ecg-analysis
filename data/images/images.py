import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import io as sio
import numpy as np

from ..utils import *


pl.seed_everything(42, workers=True)

def read_mat(fp):
    x = sio.loadmat(fp)
    data = np.asarray(x['val'], dtype=np.float64)
    return data

def read_header(fp):
    with open(fp, 'r') as f:
        header = f.readlines()
    header = [line.strip('\n') for line in header]
    return header

class PtbXlDataset:
    def __init__(
        self,
        images_dir,
        labels_file,
        transform=None,
    ):
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.transform = transform
        self.labels = pd.read_csv(labels_file, index_col='idx')
        self.classes = self.labels.columns.tolist()
        pl.seed_everything(42, workers=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = self.labels.iloc[idx].name.replace('.hea', '')
        fp = os.path.join(self.images_dir, f'{name}.png')
        image = Image.open(fp).convert('RGB')
        label = torch.tensor(self.labels.iloc[idx].tolist(), dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        
        return dict(image=image, label=label)


class ImageFromSignalDataset:
    def __init__(
        self,
        csv_file,
        signal_transform=None,
        image_transform=None,
        cached=False,
    ):
        self.csv_file = csv_file
        self.signal_tf = signal_transform
        self.image_tf = image_transform
        self.cached = cached

        self.prepare_data()
    
    def prepare_data(self):
        self.labels = pd.read_csv(self.csv_file, index_col='idx')
        self.classes = self.labels.columns.tolist()
        self.cached_nnis = dict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels.iloc[idx]
        lbl = torch.tensor(label.tolist(), dtype=torch.float)

        if idx in self.cached_nnis:
            nnis = self.cached_nnis[idx]
        else:
            header_file = label.name.replace('.mat', '.hea')
            mat_file = label.name.replace('.hea', '.mat')

            sample = load_mat(mat_file)
            headers = load_header_file(header_file)
            sampling_rate = int(headers[0].split(' ')[2])

            nnis = signal_to_nnis(sample, sampling_rate)
            if self.cached:
                self.cached_nnis[idx] = nnis
        
        img = poincare(nnis)

        if self.image_tf:
            img = self.image_tf(img)
        
        return {'image': img, 'label': lbl}


class ImageFromSignalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_csv, val_csv, test_csv=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        
        self.train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.01, 0.1), value=1.0),
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=224),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])

        self.test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=224, ratio=(0.95, 1.05)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])
        self.train_dataset = ImageFromSignalDataset(csv_file=self.train_csv, image_transform=self.train_tf)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        self.val_dataset = ImageFromSignalDataset(csv_file=self.val_csv, image_transform=self.test_tf)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        self.test_dataset = ImageFromSignalDataset(csv_file=self.test_csv, image_transform=self.test_tf)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False)



class PtbXlDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir, train_label, val_dir, val_label, test_dir=None, test_label=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.train_label = train_label
        self.val_dir = val_dir
        self.val_label = val_label
        self.test_dir = test_dir
        self.test_label = test_label
        pl.seed_everything(42, workers=True)
        
        self.train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.01, 0.1), value=1.0),
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=224),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])

        self.test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=224, ratio=(0.95, 1.05)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])
        self.train_dataset = PtbXlDataset(images_dir=self.train_dir, labels_file=self.train_label, transform=self.train_tf)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        self.val_dataset = PtbXlDataset(images_dir=self.val_dir, labels_file=self.val_label, transform=self.test_tf)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        if self.test_dir is None:
            return self.val_dataloader()

        self.test_dataset = PtbXlDataset(images_dir=self.test_dir, labels_file=self.test_label, transform=self.test_tf)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=False)