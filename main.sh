#!/bin/bash
#SBATCH --gpus=2

source ~/miniconda3/bin/activate torch_env

python3 main.py