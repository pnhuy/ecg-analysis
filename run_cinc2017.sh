#!/bin/sh

CUDA_VISIBLE_DEVICES=0 sh script/cinc2017_cnn1d.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2017_poincare_densenet121.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2017_poincare_resnet50.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2017_resnet1d.sh
CUDA_VISIBLE_DEVICES=0 sh script/cinc2017_tabular.sh