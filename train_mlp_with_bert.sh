#!/bin/bash

#SBATCH --job-name=shell
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --partition=P1
#SBATCH --output=slurm_output/bert/%j.out

python /home/jylab_share/jongsong/BLINK/blink/crossencoder/train_cross.py --train_batch_size=32 --gradient_accumulation_steps=8 --act_fn=softplus --decoder=False --dim_red=960 --layers=2 --learning_rate=0.0002 --top_k=64 --architecture mlp_with_bert --sampling False --hard_negative False --binary_loss False --num_train_epochs 20 --data_path=models/zeshel/top64_candidates/ --eval_batch_size 64 --lowercase --train_split 1 --anncur --adapter --bert_lr=1e-4 --resume=True --run_id=36bgkfoy