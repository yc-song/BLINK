#!/bin/bash

#SBATCH --job-name=shell
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --partition=P1
#SBATCH --output=slurm_output/mlp/%j.out
python /home/jylab_share/jongsong/BLINK/blink/crossencoder/train_cross.py --act_fn=softplus --architecture=extend_single --classification_head=linear --learning_rate=0.0001 --n_heads=8 --num_layers=4 --num_train_epochs=50 --optimizer=SGD --sum_dot=True --train_batch_size=4 --eval_batch_size=4 --train_split=1
 # special tokens (train size 조정)
# wandb agent jongsong/BLINK-blink_crossencoder/848ggawn
