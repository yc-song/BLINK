#!/bin/bash

#SBATCH --job-name=2
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=400000MB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.

python blink/crossencoder/train_cross.py  --train_batch_size=4 --act_fn=softplus --train_size 100 --valid_size 5000 --decoder=True --dim_red=768 --layers=4 --learning_rate=1e-3 --top_k=64  --architecture mlp_with_som --sampling False --hard_negative False --binary_loss False --num_train_epochs 10 --data_path=models/zeshel_test/top64_candidates/ --eval_batch_size 32 --lowercase --train_split 2