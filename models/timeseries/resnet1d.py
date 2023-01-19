import os

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from sklearn import metrics
import pandas as pd
from torchmetrics.classification import BinaryF1Score

import warnings
warnings.filterwarnings("ignore")


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 256, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model


class ResNet1DLightningModule(pl.LightningModule):
    def __init__(self, classes, class_weights, learning_rate):
        super().__init__()
        self.classes = classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.learning_rate = learning_rate
        pl.seed_everything(42, workers=True)
        
        self.model = resnet18(input_channels=1, inplanes=64, num_classes=len(classes))
        # self.model = ResNet1d(n_feature_maps=64, n_classes=len(classes))
        # self.model = ResNet1D(
        #     in_channels=1,
        #     base_filters=256,
        #     kernel_size=5,
        #     stride=1,
        #     groups=1,
        #     n_block=10,
        #     n_classes=len(classes),
        #     downsample_gap=2,
        #     increasefilter_gap=4,
        #     use_bn=True,
        #     use_do=True,
        #     verbose=False
        # )
        # self.model = ResNet1D(
        #     in_channels=1,
        #     base_filters=64,
        #     ratio=1.0,
        #     filter_list = [64, 160, 160, 400, 400, 1024, 1024],
        #     m_blocks_list = [2, 2, 2, 3, 3, 4, 4],
        #     kernel_size=16,
        #     stride=2,
        #     groups_width=16,
        #     verbose=False,
        #     n_classes=len(classes)
        # )

        self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_weights)
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = WeightedMultilabel(weights=self.class_weights)

        # Config metrics
        self.f1 = {
            "train": BinaryF1Score(),
            "val": BinaryF1Score(),
            "test": BinaryF1Score()
        }

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler_lr,
                "monitor": "val_loss",
            },
        }

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, stage='train'):
        data = batch['data']
        label = batch['label']
        f1 = self.f1[stage].to(self.device)

        logits = self.forward(data)
        probs = torch.sigmoid(logits)
        loss = self.loss_fn(logits, label)
        # probs = torch.softmax(logits, dim=1)
        # loss = self.loss_fn(probs, label)
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

        np.savez_compressed(os.path.join(self.logger.log_dir, stage), all_probs)

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

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, stage='train')

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
