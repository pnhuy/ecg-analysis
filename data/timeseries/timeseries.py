import os
import scipy.io as sio
import biosppy as bp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn import preprocessing
from tqdm.auto import tqdm


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, label_file, max_len=5000):
        self.data_dir = data_dir
        self.csv_file = label_file
        self.max_len = max_len
        pl.seed_everything(42, workers=True)
        
        self.prepare_data()

    def prepare_data(self):
        self.labels = pd.read_csv(self.csv_file, index_col='idx')
        self.classes = self.labels.columns.tolist()
        self.class_weights = 1.0 / np.log(self.labels.sum(axis=0))
        self.cache = dict()
        for i in tqdm(range(len(self.labels))):
            data = self.read_data_2(i)
            self.cache[i] = data

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
    
    def read_data_2(self, idx):
        # Read label
        row = self.labels.iloc[idx]
        label = torch.tensor(row.tolist(), dtype=torch.float)

       # Read data
        fp = os.path.join(
            self.data_dir, row.name
        ).replace('.hea', '.mat')
        mat = sio.loadmat(fp)

        header_fp = fp.replace('.mat', '.hea')
        if not header_fp.endswith('.hea'):
            header_fp += '.hea'
        with open(header_fp) as f:
            headers = f.readlines()

        mdata = mat['val']

        #print(mdata.shape)
        nd = np.asarray([mdata]).flatten()
        sampling_rate = int(headers[0].split(' ')[2])

        try:
            out = bp.signals.ecg.ecg(signal=nd.astype(float), sampling_rate=sampling_rate, show=False)
            ot = np.asarray(out[1])
        except ValueError:
            ot = nd.astype(float)

        length = ot.shape[0]
        #print("length of filtered signal is", length)
        maxLen = 18286

        if (length < maxLen):
            diff = maxLen - length
            ap = np.concatenate([ot, np.zeros(diff)])
        else:
            ap = ot[0 : maxLen]

        # print(ap.shape[0])
        cPD = pd.DataFrame(ap)

        la = cPD.diff()
        la = la.transpose()
        #print (la.shape)
        X = la.values.astype(np.float32)

        ## Set NaNs to 0
        X[np.isnan(X)] = 0

        X_train = preprocessing.scale(X, axis=1)
        # X_train = X / 1000.0

        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_train = torch.tensor(X_train, dtype=torch.float).reshape(1, -1)


        return dict(data=X_train, label=label, fp=fp)


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()

        # return self.read_data_2(index)
        return self.cache[index]

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
        pl.seed_everything(42, workers=True)

        self.train_dataset = TimeSeriesDataset(train_dir, train_label, max_len=max_len)
        self.val_dataset = TimeSeriesDataset(val_dir, val_label, max_len=max_len)
        if test_label:
            self.test_dataset = TimeSeriesDataset(test_dir, test_label, max_len=max_len)

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            # pin_memory=True,
            # persistent_workers=True,
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
