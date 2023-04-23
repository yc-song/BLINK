#!/bin/bash

#SBATCH --job-name=eval_bi
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=400000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/biencoder_wiki_large.bin \
# --data_path data/zeshel/blink_format --output_path models/zeshel_128 --encode_batch_size 128 --eval_batch_size 1 \
# --top_k 128 --save_topk_result --bert_model bert-large-uncased --mode valid --zeshel True --data_parallel \
# --cand_encode_path data/zeshel/cand_enc/cand_enc_valid.pt --cand_pool_path data/zeshel/cand_pool/cand_pool_valid.pt --cand_cls_path data/zeshel/cand_enc/cand_enc_valid_cls.pt

# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/biencoder_wiki_large.bin \
# --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 128 --eval_batch_size 1 \
# --top_k te --save_topk_result --bert_model bert-large-uncased --mode test --zeshel True --data_parallel \
# --cand_encode_path data/zeshel/cand_enc/cand_enc_test.pt --cand_pool_path data/zeshel/cand_pool/cand_pool_test.pt --cand_cls_path data/zeshel/cand_enc/cand_enc_test_cls.pt
# base+valid
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel/w_scores_2048 --encode_batch_size 128 --eval_batch_size 32 --top_k 2048 --save_topk_result --bert_model bert-base-cased --mode valid --zeshel True --data_parallel True --architecture special_tokens --cand_encode_path data/zeshel/cand_enc_base/cand_enc_valid.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_valid.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_valid_cls.pt --split 5
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel_test --encode_batch_size 512 --eval_batch_size 128 --top_k 64 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture mlp_with_bert --cand_pool_path data/zeshel/cand_pool --cand_encode_path data/zeshel/cand_enc --save_topk_result 
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/anncur/results/dual_encoder_zeshel.ckpt --data_path data/zeshel/blink_format --output_path models/zeshel_anncur --encode_batch_size 256 --eval_batch_size 64 --top_k 64 --bert_model bert-base-cased --mode train --zeshel True --data_parallel True --architecture mlp_with_bert --save_topk_result --anncur --lowercase --cand_pool_path data/zeshel/cand_pool --cand_encode_path data/zeshel/cand_enc 
# python blink/biencoder/eval_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel_anncur --encode_batch_size 256 --eval_batch_size 64 --top_k 64 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture mlp --lowercase --anncur --path_to_model=/home/jongsong/BLINK/dual_encoder_zeshel.ckpt --save_topk_result
# python blink/biencoder/eval_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel_anncur --encode_batch_size 256 --eval_batch_size 64 --top_k 64 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture mlp_with_bert --lowercase --anncur --path_to_model=/home/jongsong/BLINK/dual_encoder_zeshel.ckpt --save_topk_result
python blink/biencoder/eval_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel_anncur --encode_batch_size 256 --eval_batch_size 16 --top_k 64 --bert_model bert-base-cased --mode train --zeshel True --data_parallel True --architecture mlp_with_som --lowercase --anncur --path_to_model=/home/jongsong/BLINK/dual_encoder_zeshel.ckpt --save_topk_result --cand_pool_path data/zeshel/cand_pool --cand_encode_path data/zeshel/cand_enc --split 5
python blink/biencoder/eval_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel_anncur --encode_batch_size 64 --eval_batch_size 16 --top_k 64 --bert_model bert-base-cased --mode valid,test --zeshel True --data_parallel True --architecture mlp_with_som --lowercase --anncur --path_to_model=/home/jongsong/BLINK/dual_encoder_zeshel.ckpt --save_topk_result --cand_pool_path data/zeshel/cand_pool --cand_encode_path data/zeshel/cand_enc 


# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel_test --encode_batch_size 512 --eval_batch_size 32 --top_k 200 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture mlp --cand_pool_path data/zeshel/cand_pool --cand_encode_path data/zeshel/cand_enc --save_topk_result 
# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel_test --encode_batch_size 256 --eval_batch_size 128 --top_k 64 --bert_model bert-base-cased --mode valid,train,test --zeshel True --data_parallel True --architecture mlp --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool --save_topk_result 
# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel_test --encode_batch_size 256 --eval_batch_size 128 --top_k 64 --bert_model bert-base-cased --mode valid,train,test --zeshel True --data_parallel True --architecture mlp_with_bert --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool --save_topk_result 

# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 256 --eval_batch_size 128 --top_k 200 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture mlp --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool --save_topk_result 
# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 256 --eval_batch_size 128 --top_k 200 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture raw_context_text --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool --save_topk_result 

# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 256 --eval_batch_size 32 --top_k 200 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture raw_context_text --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool --save_topk_result 

# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 32 --eval_batch_size 32 --top_k 64 --save_topk_result --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture baseline --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel/w_scores_64 --encode_batch_size 256 --eval_batch_size 32 --top_k 64 --bert_model bert-base-cased --mode test --zeshel True --data_parallel True --architecture baseline --cand_encode_path data/zeshel/cand_enc_base/cand_enc_test.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_test.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_test_cls.pt --save_topk_result 
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel/w_scores_64 --encode_batch_size 256 --eval_batch_size 32 --top_k 64 --bert_model bert-base-cased --mode valid --zeshel True --data_parallel True --architecture baseline --cand_encode_path data/zeshel/cand_enc_base/cand_enc_valid.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_valid.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_valid_cls.pt --save_topk_result 
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel/w_scores_64 --encode_batch_size 256 --eval_batch_size 32 --top_k 64 --bert_model bert-base-cased --mode train --zeshel True --data_parallel True --architecture baseline --cand_encode_path data/zeshel/cand_enc_base/cand_enc_train.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_train.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_train_cls.pt --save_topk_result  
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel/w_scores_256 --encode_batch_size 128 --eval_batch_size 32 --top_k 256 --bert_model bert-base-cased --mode vaild --zeshel True --data_parallel True --architecture special_tokens --cand_encode_path data/zeshel/cand_enc_base/cand_enc_valid.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_valid.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_valid_cls.pt --save_topk_result
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel/w_scores_256 --encode_batch_size 128 --eval_batch_size 32 --top_k 256 --bert_model bert-base-cased --mode test --zeshel True --data_parallel True --architecture special_tokens --cand_encode_path data/zeshel/cand_enc_base/cand_enc_test.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_test.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_test_cls.pt --save_topk_result

# # base+test
# python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin \
# --data_path data/zeshel/blink_format --output_path models/zeshel_base_special_tokens --encode_batch_size 128 --eval_batch_size 128 \
# --top_k 64 --save_topk_result --bert_model bert-base-cased --mode valid --zeshel True --data_parallel True --architecture raw_context_text \
# --cand_encode_path data/zeshel/cand_enc_base/valid.pt --cand_pool_path data/zeshel/cand_pool_base/cand_pool_valid.pt --cand_cls_path data/zeshel/cand_enc_base/cand_enc_valid_cls.pt
# python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/biencoder_wiki_large.bin \
# --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 128 --eval_batch_size 1 \
# --top_k 64 --save_topk_result --bert_model bert-large-uncased --mode test --zeshel True --data_parallel \
# --cand_encode_path data/zeshel/cand_enc/cand_enc_test.pt --cand_pool_path data/zeshel/cand_pool/cand_pool_test.pt --cand_cls_path data/zeshel/cand_enc/cand_enc_test_cls.pt

