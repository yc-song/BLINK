#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-04:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/jongsong/.bashrc
conda activate blink


PYTHONPATH=.

# python blink/biencoder/train_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 16 --eval_batch_size 16 --bert_model bert-large-uncased --type_optimization all_encoder_layers --data_parallel


# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 128 --eval_batch_size 1 --top_k 64 --save_topk_result --bert_model bert-large-uncased --mode train,valid,test --zeshel True --data_parallel --cand_encode_path data/zeshel/cand_enc/cand_enc.pt --cand_pool_path data/zeshel/cand_pool/cand_pool.pt


python blink/crossencoder/train_cross.py --data_path  models/zeshel/top64_candidates/ --output_path models/zeshel/crossencoder --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 2 --bert_model bert-large-uncased --type_optimization all_encoder_layers --add_linear --data_parallel --zeshel True