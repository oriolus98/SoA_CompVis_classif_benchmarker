#!/bin/bash
#SBATCH --gpus=1

source ~/miniconda3/bin/activate reciclai2

python3 main.py