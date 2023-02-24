#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/%j.out

source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.

##mlp (64 candidates)
# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --dim_red=1536 --layers=4 --learning_rate=0.00032616300223161846 --train_batch_size=256 --output_path=models/zeshel/crossencoder/$SLURM_JOB_ID 
# python /home/jongsong/BLINK/blink/crossencoder/train_cross_no64.py --dim_red=1536 --layers=4 --learning_rate=0.00010049970264599704 --train_batch_size=64 --run_id xqkm59ak
##mlp (128 candidates)
# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --data_path models/zeshel/top128_candidates/ --learning_rate 1e-03 --num_train_epochs 1000 --max_context_length 128 --max_cand_length 128 --train_batch_size 512 --eval_batch_size 512 --type_optimization all_encoder_layers --data_parallel --zeshel True --output_path models/zeshel/crossencoder/2layers_adam --image_path 2layers_dim1024_lr-5 --patience 10 --layers 6 --eval_interval 200 --optimizer RMSprop --wandb semi-crossencoder --save True --dim_red 512 --train_size 10000 --valid_size 10000 --bert_model mlp --scheduler_gamma 1

## special tokens
python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --resume True --run_id $1 --learning_rate 1e-05 --num_train_epochs 1000 --train_batch_size 64 --eval_batch_size 64 --wandb "BERT with Speical Tokens" --save True --train_size 49275 --valid_size 10000 --architecture special_token --add_linear True  
sbatch train_with_run_name.sh $1


### context + special token
# python blink/crossencoder/train_cross.py --data_path  --learning_rate 1e-05 --num_train_epochs 1000 --train_batch_size 2 --eval_batch_size 4 --wandb "raw_context_text" --save True --train_size 49275 --valid_size 10000 --architecture raw_context_text --add_linear True
# sbatch train_with_run_name.sh $1