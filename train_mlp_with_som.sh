#!/bin/bash

#SBATCH --job-name=som
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=320000MB
#SBATCH --cpus-per-task=16
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
python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --architecture=mlp_with_som --decoder=False --dim_red=768 --layers=4 --learning_rate=0.001576608022167138 --num_train_epochs=10 --train_batch_size=8 --train_size 10 --valid_size 10
