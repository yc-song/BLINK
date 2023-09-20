#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --gres=gpu:4
#SBATCH --partition=P1
#SBATCH --time=0-12:00:00
#SBATCH --mem=320000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
export PYTHONPATH=$PYTHONPATH:. 

python blink/biencoder/eval_biencoder.py --data_path data/zeshel/blink_format --output_path /shared/s3/lab07/jongsong/BLINK/models/zeshel --encode_batch_size 256 --eval_batch_size 64 --top_k 1024 --bert_model bert-base-uncased --mode train --zeshel True --data_parallel True --architecture mlp_with_bert --path_to_model=./retriever.bin --split 1 --cand_pool_path data/zeshel/cand_pool_bert --cand_encode_path data/zeshel/cand_enc_mvd --save_topk_nce
# python blink/crossencoder/evaluate_cross.py --act_fn=softplus --architecture=baseline --classification_head=dot --learning_rate=0.0001220800198325272 --n_heads=2 --num_layers=6 --num_train_epochs=100 --optimizer=SGD --train_batch_size=32 --data_path=/shared/s3/lab07/jongsong/BLINK/models/zeshel/top64_candidates --train_split 1 --patience=20 --top_k=64 --identity_init true --bert_model=bert-base-uncased --eval_batch_size=32 --add_linear True --path_to_model=/shared/s3/lab07/jongsong/hard-nce-el/models/full/anncur/cls_crossenc_zeshel.ckpt --anncur
