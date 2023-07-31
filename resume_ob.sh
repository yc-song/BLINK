#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=slurm_output/multi/%j.out
#SBATCH --error=slurm_output/multi/%j.error

python blink/crossencoder/train_cross.py --wandb=self-attention --resume=True --run_id=6x49j21s --act_fn=softplus --architecture=extend_multi --classification_head=dot --learning_rate=1e-05 --n_heads=2 --num_layers=2 --num_train_epochs=500 --optimizer=SGD --train_batch_size=512 --train_split=1
