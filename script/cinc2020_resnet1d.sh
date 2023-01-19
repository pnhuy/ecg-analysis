#!/bin/sh

python train_resnet1d.py --csv_path dataset/cinc2020/processed --data_path dataset/cinc2020/raw/ --log_dir logs_cinc2020/resnet1d --max_epoch 100