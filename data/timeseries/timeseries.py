import os
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, label_file, max_len=5000):
        self.data_dir = data_dir
        self.csv_file = label_file
        self.max_len = max_len
        
        self.prepare_data()

    def prepare_data(self):
        self.labels = pd.read_csv(self.csv_file, index_col='idx')
        self.classes = self.labels.columns.tolist()
        self.class_weights = 1.0 / np.log(self.labels.sum(axis=0))

    def _pad(self, x):
        if len(x) >= self.max_len:
            return x[:self.max_len]
        
        pad_width = self.max_len - len(x)
        x = np.pad(x, pad_width=(0, pad_width))
        
        return x

    def read_data(self, idx):
        # Read label
        row = self.labels.iloc[idx]
        label = torch.tensor(row.tolist(), dtype=torch.float)

        # Read data
        fp = os.path.join(
            self.data_dir, row.name
        ).replace('.hea', '.mat')
        
        mat = sio.loadmat(fp)
        mat = np.array(mat['val'])[0] / 1000.0
        mat = self._pad(mat)
        mat = torch.tensor(mat, dtype=torch.float).unsqueeze(0)

        return dict(data=mat, label=label)

    def __getitem__(self, index):
        return self.read_data(index)

    def __len__(self):
        return len(self.labels)


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, train_label, val_dir, val_label, test_dir=None, test_label=None, batch_size=16, max_len=3000):
        super().__init__()
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.train_label = train_label
        self.val_dir = val_dir
        self.val_label = val_label
        self.test_dir = test_dir
        self.test_label = test_label

        self.train_dataset = TimeSeriesDataset(train_dir, train_label, max_len=max_len)
        self.val_dataset = TimeSeriesDataset(val_dir, val_label, max_len=max_len)
        if test_label:
            self.test_dataset = TimeSeriesDataset(test_dir, test_label, max_len=max_len)

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False
        )
