#!/bin/bash

#SBATCH --job-name=bert
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.

##mlp (64 candidates)
python_output=$(python blink/crossencoder/train_cross.py --learning_rate 2e-05 --num_train_epochs 20 --train_batch_size 256 --eval_batch_size 1024 --wandb "BERT with Speical Tokens" --save True --train_size -1 --valid_size -1 --architecture special_token --add_linear True)
time_out=$(echo "$python_output" | grep -E "time_out")
run_id=$(echo "$python_output" | grep -Eo '(^|[ ,])run_id:[^,]*' | cut -d: -f2-)
if [ "$time_out" = "time_out" ]; then
  echo "timeout"
  sbatch resume_bert.sh $run_id
else
  echo "Python script ran successfully"
  # sbatch train_bert.sh
fi