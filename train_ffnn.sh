#!/bin/bash

#SBATCH --job-name=ffnn
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/ffnn/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/ffnn/%j.error
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
wandb enabled
wandb login
python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --decoder=False --dim_red=1152 --layers=2 --learning_rate=0.00016843606494416268 --train_batch_size=416 --train_split 1

