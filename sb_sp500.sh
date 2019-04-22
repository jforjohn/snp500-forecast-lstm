#!/bin/bash
#SBATCH --job-name="sp500_mvar_stat_job"
#SBATCH --workdir=.
#SBATCH --output=sp500_%j.out
#SBATCH --error=sp500_%j.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=01:59:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python run.py
