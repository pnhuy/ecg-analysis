#!/bin/sh

CUDA_VISIBLE_DEVICES=0 sh script/cinc2020_cnn1d.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2020_poincare_densenet121.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2020_poincare_resnet50.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2020_resnet1d.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2020_tabular.sh