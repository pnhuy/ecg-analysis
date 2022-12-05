import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = self.labels.iloc[idx].name
        fp = os.path.join(self.images_dir, f'{name}.png')
        image = Image.open(fp).convert('RGB')
        label = torch.tensor(self.labels.iloc[idx].tolist(), dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        
        return dict(image=image, label=label)


class PtbXlDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir, train_label, val_dir, val_label, test_dir=None, test_label=None):
        super().__init__()
        self.batch_size = batch_size
        
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.01, 0.1), value=1.0),
            transforms.Resize(size=224),
            # transforms.RandomResizedCrop(size=224),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=224),
            # transforms.RandomResizedCrop(size=224, ratio=(0.95, 1.05)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ])

        self.train_dataset = PtbXlDataset(images_dir=train_dir, labels_file=train_label, transform=train_tf)
        self.val_dataset = PtbXlDataset(images_dir=val_dir, labels_file=val_label, transform=test_tf)
        if test_dir:
            self.test_dataset = PtbXlDataset(images_dir=test_dir, labels_file=test_label, transform=test_tf)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=False)