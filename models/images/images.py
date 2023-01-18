import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch import optim
from torch.optim import lr_scheduler
from torchmetrics.classification import BinaryF1Score
import numpy as np
from sklearn import metrics
import pandas as pd
from torchvision.ops.focal_loss import sigmoid_focal_loss

import pytorch_lightning as pl


pl.seed_everything(42, workers=True)

class ImageClassifier(pl.LightningModule):
    def __init__(self, classes=[], barebone='resnet50', learning_rate=1e-3, loss_type='bce'):
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.barebone = barebone
        self.learning_rate=learning_rate
        self.loss_type = loss_type
        pl.seed_everything(42, workers=True)

        if loss_type == 'bce':
            self.loss_fn = F.binary_cross_entropy_with_logits
        if loss_type == 'focal':
            self.loss_fn = sigmoid_focal_loss

        model_fn = getattr(models, barebone)
        self.network = model_fn(weights=None)
        
        if barebone.startswith('vit'):
            self.network.heads[0] = nn.Linear(self.network.heads[0].in_features, self.num_classes)
        elif barebone.startswith('resnet'):
            self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)
        elif barebone.startswith('densenet'):
            self.network.classifier = nn.Linear(self.network.classifier.in_features, self.num_classes)

        # Config metrics
        self.f1 = {
            "train": BinaryF1Score(),
            "val": BinaryF1Score(),
            "test": BinaryF1Score()
        }
    
    def configure_optimizers(self):
        # self.optimizer = optim.AdamW(self.network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        # self.scheduler_lr = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
        # self.scheduler_lr = lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate, # cycle_momentum=False, 
        #                                           max_lr=self.learning_rate*10, step_size_up=5, mode="triangular2")
        # return [self.optimizer], [self.scheduler_lr]

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler_lr,
                "monitor": "val_loss",
            },
        }

    def forward(self, image):
        out = self.network(image)
        return out
    
    def _step(self, batch, batch_idx, stage='train'):
        image= batch['image']
        label = batch['label']
        bs = image.shape[0]
        f1 = self.f1[stage].to(self.device)
        
        preds = self.forward(image)
        logits = preds # preds.logits if stage == 'train' and else preds
        probs = torch.sigmoid(logits)

        loss = self.loss_fn(logits, label, reduction='mean')
        
        f1.update(probs, label)

        self.log(f'{stage}_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=image.shape[0], sync_dist=True)
        
        if stage == 'train':
            current_lr = self.optimizer.param_groups[0]['lr'] # self.scheduler_lr.get_last_lr()[0]
            self.log('learning_rate', current_lr, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        logger = self.logger.experiment
        if (batch_idx == 0) and self.current_epoch < 10:
            img_grid = torchvision.utils.make_grid(image, nrow=bs//int(bs**0.5))
            logger.add_image(f'{stage}_image', img_grid, self.current_epoch)
        
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
        elif stage == 'train':
            return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, stage='val')
    
    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, stage='test')
    
    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, stage='train')