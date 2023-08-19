#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --gres=gpu:4
#SBATCH --partition=P1
#SBATCH --nodelist=a02
#SBATCH --time=0-12:00:00
#SBATCH --mem=320000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
export PYTHONPATH=$PYTHONPATH:. 
 python blink/biencoder/eval_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 256 --eval_batch_size 64 --top_k 1024 --bert_model bert-base-uncased --mode valid,test --zeshel True --data_parallel True --architecture extend_multi --path_to_model=./dual_encoder_zeshel.ckpt --split 1 --save_topk_result --anncur --cand_pool_path data/zeshel/cand_pool_bert --cand_encode_path data/zeshel/cand_enc_bert --save_topk_result
