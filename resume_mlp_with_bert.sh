#!/bin/bash
#SBATCH --job-name=bert-resume
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error
#SBATCH --mem=320000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
## usage: resume_ffnn.sh job_id
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
lr=`grep -o '"learning_rate": [^,]*' ./models/zeshel/crossencoder/mlp_with_bert/$1/training_params/training_params.json | grep -o '[^ ]*$'`
num_train_epochs=`grep -o '"num_train_epochs": [^,]*' ./models/zeshel/crossencoder/mlp_with_bert/$1/training_params/training_params.json | grep -o '[^ ]*$'`s
wandb enabled
wandb online
# echo "lr=${lr}"
python blink/crossencoder/train_cross.py --train_batch_size=4 --gradient_accumulation_steps=2 --act_fn=softplus --decoder=True --dim_red=1152 --layers=2 --learning_rate=0.00025183381898432147 --top_k=64 --architecture mlp_with_bert --sampling False --hard_negative False --binary_loss False --num_train_epochs 10 --data_path=models/zeshel_test/top64_candidates/ --path_to_model models/zeshel/biencoder/pytorch_model.bin --eval_batch_size 32 --lowercase --train_split 1 --path_to_mlpmodel models/zeshel/crossencoder/mlp/q0efm6j0/epoch_88_0 --resume=True --run_id=mpahac56
# echo $python_output
# time_out=$(echo "$python_output" | grep -E "time_out")
# if [ "$time_out" = "time_out" ]; then
#   echo "timeout"
#   sbatch resume_raw_bert.sh $1
# else
#   echo "Python script ran successfully"
#   # sbatch train_raw_bert.sh
# fi