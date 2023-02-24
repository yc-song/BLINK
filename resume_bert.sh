#!/bin/bash
#SBATCH --job-name=bert-resume
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
## usage: resume_ffnn.sh job_id
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
lr=`grep -o '"learning_rate": [^,]*' ./models/zeshel/crossencoder/$1/training_params/training_params.json | grep -o '[^ ]*$'`
num_train_epochs=`grep -o '"num_train_epochs": [^,]*' ./models/zeshel/crossencoder/$1/training_params/training_params.json | grep -o '[^ ]*$'`

echo "lr=${lr}"
python_output=$(python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --learning_rate=${lr}  --resume=True --run_id=$1 --num_train_epochs=${num_train_epochs} --train_batch_size 512 --eval_batch_size 1024 --wandb "BERT with Speical Tokens" --save True --train_size -1 --valid_size -1 --architecture special_token --add_linear True)
echo $python_output
time_out=$(echo "$python_output" | grep -E "time_out")
if [ "$time_out" = "time_out" ]; then
  echo "timeout"
  sbatch resume_bert.sh $1
else
  echo "Python script ran successfully"
  # sbatch train_bert.sh
fi