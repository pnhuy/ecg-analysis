import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from data.images import PtbXlDataModule
from models.images import ImageClassifier
import os


def set_seed(seed=0):
    import numpy, torch, random
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_barebone', type=str, default='resnet50')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def train(args):
    set_seed(seed=args.seed)

    train_dir = os.path.join(args.data_path, 'processed')
    train_label = os.path.join(args.data_path, 'processed/y_train.csv')
    val_dir = os.path.join(args.data_path, 'processed')
    val_label = os.path.join(args.data_path, 'processed/y_val.csv')
    test_dir = os.path.join(args.data_path, 'processed')
    test_label = os.path.join(args.data_path, 'processed/y_test.csv')

    logger = TensorBoardLogger(args.log_dir)
    
    datamodule = PtbXlDataModule(
        train_dir=train_dir,
        train_label=train_label,
        val_dir=val_dir,
        val_label=val_label,
        # test_dir=test_dir,
        # test_label=test_label,
        batch_size=args.batch_size
    )

    classes = datamodule.train_dataset.labels.columns
    print('Train data lenghth:', len(datamodule.train_dataset))
    model = ImageClassifier(
        classes=classes,
        barebone=args.model_barebone, # 'vit_b_16'
        learning_rate=args.learning_rate,
        loss_type='bce'
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, 'ckpt'),
        mode='min',
        monitor='val_loss',
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
        save_last=True,
        save_top_k=-1,
        every_n_epochs=10,
    )

    best_ckpt = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, 'ckpt'),
        mode='min',
        monitor='val_loss',
        filename='best-{epoch}-{val_loss:.2f}-{val_f1:.2f}',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint, best_ckpt]
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )


if __name__ == '__main__':
    args = parse_args()
    train(args)
