"""
Pytorch Version of Ruhi solution of CinC 2017
"""

import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl
from sklearn import metrics
import pandas as pd
from torchmetrics.classification import BinaryF1Score

class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        # import pdb; pdb.set_trace()
        device = outputs.device
        self.criterion = self.criterion.to(device)
        self.weights = self.weights.to(device)
        loss = self.criterion(outputs, targets)
        return (loss * self.weights).mean()


class Net1DLightningModule(pl.LightningModule):
    def __init__(self, classes, class_weights):
        super().__init__()
        self.classes = classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=20, stride=1, padding="valid"),
            nn.BatchNorm1d(256, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding="valid"),
            nn.BatchNorm1d(256, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding="valid"),
            nn.BatchNorm1d(128, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding="valid"),
            nn.BatchNorm1d(128, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding="valid"),
            nn.BatchNorm1d(128, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding="valid"),
            nn.BatchNorm1d(128, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm1d(64, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm1d(64, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm1d(64, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm1d(64, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm1d(32, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm1d(32, momentum=0.99),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.AvgPool1d(kernel_size=1, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=4)
        )

        self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_weights)
        # self.loss_fn = WeightedMultilabel(weights=self.class_weights)

        # Config metrics
        self.f1 = {
            "train": BinaryF1Score(),
            "val": BinaryF1Score(),
            "test": BinaryF1Score()
        }

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        # return {
        #     "optimizer": self.optimizer,
        #     "lr_scheduler": {
        #         "scheduler": self.scheduler_lr,
        #         "monitor": "val_loss",
        #     },
        # }
        return self.optimizer

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, stage='train'):
        data = batch['data']
        label = batch['label']
        f1 = self.f1[stage].to(self.device)

        logits = self.forward(data)
        probs = torch.sigmoid(logits)
        loss = self.loss_fn(logits, label)
        f1.update(probs, label)

        self.log(f'{stage}_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=data.shape[0])

        if stage == 'train' and batch_idx == 0:
            current_lr = self.optimizer.param_groups[0]['lr'] # self.scheduler_lr.get_last_lr()[0]
            self.log('learning_rate', current_lr, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return dict(
            loss=loss,
            probs=probs.detach().cpu().numpy(),
            labels=label.detach().cpu().numpy()
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='test')
    
    def _epoch_end(self, outputs, stage='val'):
        logger = self.logger.experiment

        all_probs = np.concatenate([i['probs'] for i in outputs])
        all_labels = np.concatenate([i['labels'] for i in outputs])
        all_preds = np.where(all_probs > 0.5, 1, 0)

        # Generate classification report and log to tensorboard
        report = metrics.classification_report(all_labels, all_preds, 
            target_names=self.classes, zero_division=0, output_dict=True)
        report = pd.DataFrame(report).round(2).T.to_markdown()
        logger.add_text(f'{stage}_clf_report', report, self.current_epoch)

        if stage == 'val':
            return super().validation_epoch_end(outputs)
        elif stage == 'test':
            return super().test_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, stage='val')
    
    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, stage='test')

    def _on_epoch_end(self, stage='train'):
        f1 = self.f1[stage]
        score = f1.compute()
        f1.reset()
        self.log(f'{stage}_f1', score, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_train_epoch_end(self):
        return self._on_epoch_end(stage='train')

    def on_validation_epoch_end(self):
        return self._on_epoch_end(stage='val')
    
    def on_test_epoch_end(self):
        return self._on_epoch_end(stage='test')