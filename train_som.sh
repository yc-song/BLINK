#!/bin/bash

#SBATCH --job-name=som
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
wandb enabled
python blink/crossencoder/evaluate_cross.py --train_batch_size=32 --act_fn=softplus --decoder=True --dim_red=768 --layers=4 --learning_rate=1e-5 --top_k=64 --architecture som --sampling False --hard_negative False --binary_loss False --num_train_epochs 20 --data_path=models/zeshel_test/top64_candidates/ --eval_batch_size 64 --train_size 10 --valid_size 10 --lowercase