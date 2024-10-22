#!/bin/bash
#SBATCH --gpus=2

source ~/miniconda3/bin/activate CV_bench_env

python3 main.py