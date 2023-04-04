#!/bin/bash

#SBATCH --job-name=mlp-with-som
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=240000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error

source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink

PYTHONPATH=.
wandb enabled
wandb online
wandb agent jongsong/mlp-with-som/ves3748u --count 1
sbatch train_sweep_mlp_with_som.sh
