#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=P1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --job-name=x425ydx5
#SBATCH --output=slurm_output/bert/%A-%a.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
## usage: sbatch sbatch_bert.sh [lr]
bash srun_bert.sh $1