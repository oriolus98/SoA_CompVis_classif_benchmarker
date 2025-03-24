#!/bin/bash
#SBATCH --gpus=2

source ~/miniconda3/bin/activate reciclai2

python3 agg_pred.py