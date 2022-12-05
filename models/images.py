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

class ImageClassifier(pl.LightningModule):
    def __init__(self, classes=[], barebone='resnet50', learning_rate=1e-3, loss_type='bce'):
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.barebone = barebone
        self.learning_rate=learning_rate
        self.loss_type = loss_type

        if loss_type == 'bce':
            self.loss_fn = F.binary_cross_entropy_with_logits
        if loss_type == 'focal':
            self.loss_fn = sigmoid_focal_loss

        model_fn = getattr(models, barebone)
        self.network = model_fn(weights=None)
        if barebone.startswith('vit'):
            self.network.heads[0] = nn.Linear(self.network.heads[0].in_features, self.num_classes)
        else:
            self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)

        # Config metrics
        self.f1 = {
            "train": BinaryF1Score(),
            "val": BinaryF1Score(),
            "test": BinaryF1Score()
        }

    
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.learning_rate)
        self.scheduler_lr = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
        return [self.optimizer], [self.scheduler_lr]

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
            self.log('learning_rate', self.scheduler_lr.get_last_lr()[0], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

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

    def validation_epoch_end(self, outputs):
        logger = self.logger.experiment

        all_probs = np.concatenate([i['probs'] for i in outputs])
        all_labels = np.concatenate([i['labels'] for i in outputs])
        all_preds = np.where(all_probs > 0.5, 1, 0)

        # Generate classification report and log to tensorboard
        report = metrics.classification_report(all_labels, all_preds, 
            target_names=self.classes, zero_division=0, output_dict=True)
        report = pd.DataFrame(report).round(2).T.to_markdown()
        logger.add_text('val_clf_report', report, self.current_epoch)

        return super().validation_epoch_end(outputs)