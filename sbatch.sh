#!/bin/bash
#SBATCH --job-name=full_context
#SBATCH --output=slurm_output/%A-%a.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
# add sweep parameters
python blink/crossencoder/train_cross.py --act_fn=softplus --architecture=extend_multi --classification_head=dot --learning_rate=2e-05 --n_heads=16 --num_layers=4 --num_train_epochs=100 --optimizer=SGD --train_batch_size=256 --train_split=1