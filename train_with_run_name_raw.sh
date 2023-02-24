#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/%j.out

source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.

### context + special token
python blink/crossencoder/train_cross.py --data_path  --learning_rate 1e-05 --num_train_epochs 1000 --train_batch_size 2 --eval_batch_size 4 --wandb "raw_context_text" --save True --train_size 49275 --valid_size 10000 --architecture raw_context_text --add_linear True
sbatch train_with_run_name_raw.sh $1