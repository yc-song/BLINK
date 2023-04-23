#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=128000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/nartaech/%j.out

source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink

PYTHONPATH=.
## 6 layers
# wandb agent jongsong/BLINK-blink_crossencoder/hzf1jjgv
## 2,4 layers
# wandb agent jongsong/BLINK-blink_crossencoder/3vj735pc
## BERT-base (2,4 layers)
# wandb agent jongsong/BLINK-blink_crossencoder/5ciifeq9
# 500 epochs
# wandb agent jongsong/BLINK-blink_crossencoder/3vj735pc
# 1000 epochs
#  wandb agent jongsong/BLINK-blink_crossencoder/fyp66yyz
# 1000 epochs with batch size 64
# wandb agent jongsong/BLINK-blink_crossencoder/gi9g29ip
# special tokens
# wandb agent jongsong/BLINK-blink_crossencoder/a3oguqkg
# special tokens (scheduler 조정)
# wandb agent jongsong/BLINK-blink_crossencoder/2nhyuf56
# special tokens (batch size 조정)
# wandb agent jongsong/BLINK-blink_crossencoder/jwkkov0m

## 2022.01.22
# FFNN (more hyperparameters)
# wandb agent jongsong/BLINK-blink_crossencoder/hk6df3p3
# FFNN (w/o label idx 64)
wandb agent jongsong/BLINK-blink_crossencoder/tnjarq8r --count 20
sbatch train_sweep.sh
# special tokens (train size 조정)
# wandb agent jongsong/BLINK-blink_crossencoder/848ggawn
