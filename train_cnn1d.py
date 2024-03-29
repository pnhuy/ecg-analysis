import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import eco2ai

from data.timeseries.timeseries import TimeSeriesDataModule
from models.timeseries.cnn import Net1DLightningModule

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=0):
    import numpy, torch, random
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/huypham/Projects/ecg/dataset/cinc2020/raw')
    parser.add_argument('--csv_path', type=str, default='/home/huypham/Projects/ecg/dataset/cinc2020/processed')
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--model_barebone', type=str, default='resnet50')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./logs/cnn1d')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def train(args=None):
    set_seed(args.seed)
    train_dir = args.data_path
    val_dir = train_dir
    test_dir = train_dir

    train_label = os.path.join(args.csv_path, 'y_train.csv') # '/home/huypham/Projects/ecg/dataset/cinc2020/processed/y_train.csv'
    val_label = os.path.join(args.csv_path, 'y_val.csv')
    test_label = os.path.join(args.csv_path, 'y_test.csv')

    data_module = TimeSeriesDataModule(
        train_dir=train_dir,
        train_label=train_label,
        val_dir=val_dir,
        val_label=val_label,
        test_dir=test_dir,
        test_label=test_label,
        batch_size=args.batch_size
    )

    train_dataloader = data_module.train_dataloader()
    classes = data_module.train_dataset.classes
    class_weights = data_module.train_dataset.class_weights

    data = next(iter(train_dataloader))
    print(data['data'].shape)
    print(data['label'].shape)

    model = Net1DLightningModule(classes=classes, class_weights=class_weights)

    logger = TensorBoardLogger(args.log_dir)

    eco_tracker = eco2ai.Tracker(
        file_name=os.path.join(logger.log_dir, "emission.csv")
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'ckpt'),
        mode='min',
        monitor='val_loss',
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
        save_last=True,
        save_top_k=-1,
        every_n_epochs=10,
    )

    best_ckpt = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'ckpt'),
        mode='min',
        monitor='val_loss',
        filename='best-{epoch}-{val_loss:.2f}-{val_f1:.2f}',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint, best_ckpt],
        precision=16
    )

    eco_tracker.start()

    trainer.fit(model, data_module)

    trainer.test(model, data_module, ckpt_path=best_ckpt.best_model_path)

    eco_tracker.stop()


if __name__ == '__main__':
    args = parse_args()
    train(args)