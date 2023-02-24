#!/bin/bash

#SBATCH --job-name=train_biencoder
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=0-12:00:00
#SBATCH --mem=32000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
## BERT-base
python blink/biencoder/train_biencoder.py --optimizer AdamW --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder_base --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 128 --eval_batch_size 64 --bert_model bert-base-cased --type_optimization all_encoder_layers --data_parallel True
## BERT-large
# python blink/biencoder/train_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder  --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 64 --eval_batch_size 64 --bert_model bert-large-uncased --type_optimization all_encoder_layers --data_parallel/
## MLP
# python blink/biencoder/train_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder_mlp --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 16 --eval_batch_size 16 --bert_model bert-base-cased --type_optimization all_encoder_layers --data_parallel True --with_mlp