#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=400000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/ffnn/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/ffnn/%j.error
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --act_fn=softplus --decoder=False --dim_red=1536 --layers=6 --learning_rate=0.0004471484265568258 --train_batch_size=2
##mlp (64 candidates)
# python_output=$(python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --act_fn=softplus --decoder=False --dim_red=1536 --layers=6 --learning_rate=0.0004471484265568258 --train_batch_size=640)
# time_out=$(echo "$python_output" | grep -E "time_out")
# run_id=$(echo "$python_output" | grep -Eo '(^|[ ,])run_id:[^,]*' | cut -d: -f2-)
# if [ "$time_out" = "time_out" ]; then
#   echo "timeout"
#   sbatch resume_ffnn.sh $run_id
# else
#   echo "Python script ran successfully"
#   sbatch train.sh
# fi
# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --dim_red=1536 --layers=4 --learning_rate=0.00032616300223161846 --train_batch_size=256 --output_path=models/zeshel/crossencoder/$SLURM_JOB_ID 
# python /home/jongsong/BLINK/blink/crossencoder/train_cross_no64.py --dim_red=1536 --layers=4 --learning_rate=0.00010049970264599704 --train_batch_size=64 --run_id xqkm59ak
##mlp (128 candidates)
# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --data_path models/zeshel/top128_candidates/ --learning_rate 1e-03 --num_train_epochs 1000 --max_context_length 128 --max_cand_length 128 --train_batch_size 512 --eval_batch_size 512 --type_optimization all_encoder_layers --data_parallel --zeshel True --output_path models/zeshel/crossencoder/2layers_adam --image_path 2layers_dim1024_lr-5 --patience 10 --layers 6 --eval_interval 200 --optimizer RMSprop --wandb semi-crossencoder --save True --dim_red 512 --train_size 10000 --valid_size 10000 --bert_model mlp --scheduler_gamma 1

# ## special tokens

# output = `python blink/crossencoder/train_cross.py --learning_rate 1e-05 --num_train_epochs 1000 --train_batch_size 128 --eval_batch_size 128 --wandb "BERT with Speical Tokens" --save True --train_size 49275 --valid_size 10000 --architecture special_token --add_linear True`
# run_id=${output: (-11):8}
# sbatch train_with_run_name.sh $run_id


## context + special token
# output = `python blink/crossencoder/train_cross.py --learning_rate 1e-05 --num_train_epochs 1000 --train_batch_size 2 --eval_batch_size 4 --wandb "BERT with Speical Tokens" --save True --train_size 49275 --valid_size 10000 --architecture raw_context_text --add_linear True`
# run_id=${output: (-11):8}
# sbatch train_with_run_name_raw.sh $run_id

# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --data_path models/zeshel/top64_candidates_raw_text/ --learning_rate 1e-05 --num_train_epochs 100 --max_context_length 128 --max_cand_length 128 --train_batch_size 1 --eval_batch_size 1 --type_optimization all_encoder_layers --data_parallel --zeshel True --output_path models/zeshel/crossencoder/2layers_adam --image_path 2layers_dim1024_lr-5 --patience 10 --layers 2 --eval_interval 1000 --optimizer Adam --wandb semi-crossencoder --save True --dim_red 512 --train_size 1000 --valid_size 100 --architecture raw_context_text --scheduler_gamma 1 --add_linear
# python /home/jongsong/BLINK/blink/crossencoder/train_cross_bert.py --bert_model=bert-large-uncased --data_path models/zeshel/top64_candidates_raw_text/ --learning_rate 1e-03 --num_train_epochs 50 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 2 --type_optimization all_encoder_layers --data_parallel --zeshel True --output_path models/zeshel/crossencoder/2layers_adam --patience 10 --layers 2 --eval_interval 500 --optimizer Adam --wandb semi-crossencoder --save True --dim_red 512 --train_size 1005 --valid_size 500 --architecture raw_context_text --scheduler_gamma 1 --add_linear --data_parallel
# python /home/jongsong/BLINK/blink/crossencoder/train_cross_bert.py --bert_model=bert-base-cased --data_path models/zeshel/top64_candidates_base_raw_text/ --learning_rate 1e-04 --num_train_epochs 50 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 2 --type_optimization all_encoder_layers --data_parallel --zeshel True --output_path models/zeshel/crossencoder/2layers_adam --patience 10 --layers 2 --eval_interval 500 --optimizer Adam --wandb semi-crossencoder --save True --dim_red 512 --train_size 1005 --valid_size 500 --architecture raw_context_text --scheduler_gamma 1 --add_linear --data_parallel
# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --bert_model=bert-base-cased --data_path models/zeshel/top64_candidates_base_raw_text/ --learning_rate 1e-04 --num_train_epochs 50 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 4 --type_optimization all_encoder_layers --data_parallel --zeshel True --output_path models/zeshel/crossencoder/2layers_adam --patience 10 --layers 2 --eval_interval 500 --optimizer Adam --wandb semi-crossencoder --save True --dim_red 512 --train_size 500 --valid_size 100 --architecture raw_context_text --scheduler_gamma 1 --add_linear --data_parallel --wandb raw_context_text