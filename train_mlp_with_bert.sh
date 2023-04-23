#!/bin/bash

#SBATCH --job-name=2
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=200000MB
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
wandb online
python blink/crossencoder/train_cross.py --train_batch_size=4 --gradient_accumulation_steps=1 --act_fn=softplus --decoder=True --dim_red=1536 --layers=2 --learning_rate=0.0005043889102925416 --top_k=64 --architecture mlp_with_bert --sampling False --hard_negative False --binary_loss False --num_train_epochs 10 --data_path=models/zeshel_anncur/top64_candidates/ --path_to_model /home/jongsong/BLINK/dual_encoder_zeshel.ckpt  --eval_batch_size 32 --lowercase --train_split 1 --path_to_mlpmodel models/zeshel/crossencoder/mlp/0yairdza/Epochs/epoch_17 --anncur
