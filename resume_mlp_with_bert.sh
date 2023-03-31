#!/bin/bash
#SBATCH --job-name=bert-resume
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
## usage: resume_ffnn.sh job_id
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
lr=`grep -o '"learning_rate": [^,]*' ./models/zeshel/crossencoder/mlp_with_bert/$1/training_params/training_params.json | grep -o '[^ ]*$'`
num_train_epochs=`grep -o '"num_train_epochs": [^,]*' ./models/zeshel/crossencoder/mlp_with_bert/$1/training_params/training_params.json | grep -o '[^ ]*$'`

echo "lr=${lr}"
python_output=$(python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --data_path=models/zeshel_test/top64_candidates/ --learning_rate=${lr} --resume=True --run_id=$1 --num_train_epochs=${num_train_epochs} --train_batch_size 2 --eval_batch_size 8 --save True --train_size 10000 --valid_size 5000 --decoder=True --dim_red=768 --layers=4 --wandb mlp-with-bert --architecture mlp_with_bert)
echo $python_output
time_out=$(echo "$python_output" | grep -E "time_out")
# if [ "$time_out" = "time_out" ]; then
#   echo "timeout"
#   sbatch resume_raw_bert.sh $1
# else
#   echo "Python script ran successfully"
#   # sbatch train_raw_bert.sh
# fi