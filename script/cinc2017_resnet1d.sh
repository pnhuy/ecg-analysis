#!/bin/sh
python train_resnet1d.py --csv_path dataset/cinc2017/processed --data_path dataset/cinc2017/raw/training --log_dir logs_cinc2017/resnet1d --max_epoch 100