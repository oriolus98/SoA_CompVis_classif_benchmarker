#!/bin/bash
#SBATCH --gpus=1

source ~/miniconda3/bin/activate reciclai2

python3 agg_pred_with_yolo.py