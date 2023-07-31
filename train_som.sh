#!/bin/bash

#SBATCH --job-name=som2
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=200000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --partition=P1
#SBATCH --output=slurm_output/som/%j.out
wandb agent jongsong/mlp-with-som/udmq0mxn

