import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data.timeseries.timeseries import TimeSeriesDataModule
from models.timeseries.resnet1d import ResNet1DLightningModule

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/huypham/Projects/ecg/dataset/cinc2020/raw')
    parser.add_argument('--csv_path', type=str, default='/home/huypham/Projects/ecg/dataset/cinc2020/processed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_dir', type=str, default='./logs/resnet1d')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def train(args=None):
    pl.seed_everything(args.seed, workers=True)
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

    # import ipdb; ipdb.set_trace()

    data = next(iter(train_dataloader))
    print(data['data'].shape)
    print(data['label'].shape)
    print(data['data'].max())
    print(data['data'].min())

    model = ResNet1DLightningModule(
        classes=classes,
        class_weights=class_weights,
        learning_rate=args.learning_rate,
    )

    logger = TensorBoardLogger(args.log_dir)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'ckpt'),
        mode='min',
        monitor='val_f1',
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

    trainer.fit(model, data_module)
    
    trainer.test(model, data_module, ckpt_path=best_ckpt.best_model_path)


if __name__ == '__main__':
    args = parse_args()
    train(args)