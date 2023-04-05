#!/bin/bash
#SBATCH --job-name=wandb-resume
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jongsong/BLINK/slurm_output/ffnn/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/ffnn/%j.error
#SBATCH --mem=200000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
## usage: resume_ffnn.sh job_id
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.
act_fn=`grep -o '"act_fn": [^,]*' ./models/zeshel/crossencoder/mlp/$1/training_params/training_params.json | grep -o '[^ ]*$'`
act_fn=`echo "$act_fn" | cut -d \" -f2`
lr=`grep -o '"learning_rate": [^,]*' ./models/zeshel/crossencoder/mlp/$1/training_params/training_params.json | grep -o '[^ ]*$'`
dim_red=`grep -o '"dim_red": [^,]*' ./models/zeshel/crossencoder/mlp/$1/training_params/training_params.json | grep -o '[^ ]*$'`
layers=`grep -o '"layers": [^,]*' ./models/zeshel/crossencoder/mlp/$1/training_params/training_params.json | grep -o '[^ ]*$'`
train_batch_size=`grep -o '"train_batch_size": [^,]*' ./models/zeshel/crossencoder/mlp/$1/training_params/training_params.json | grep -o '[^ ]*$'`
decoding=`grep -o '"decoder": [^,]*' ./models/zeshel/crossencoder/mlp/$1/training_params/training_params.json | grep -o '[^ ]*$'`
echo "act_fn=${act_fn}, lr=${lr}, dim_red=${dim_red}, layers=${layers}, train_batch_size=${train_batch_size}, decoding=${decoding}"
python_output=$(python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --act_fn=${act_fn} --decoder=${decoding} --dim_red=${dim_red} --layers=${layers} --learning_rate=${lr} --train_batch_size=${train_batch_size} --sampling False --eval_batch_size 32 --hard_negative False --binary_loss False --data_path=models/zeshel_test/top1000_candidates/ --resume=True --run_id=$1 --train_split 1)
echo $python_output
time_out=$(echo "$python_output" | grep -E "time_out")
run_id=$1
if [ "$time_out" = "time_out" ]; then
  echo "timeout"
  sbatch resume_ffnn.sh $run_id
else
  echo "Python script ran successfully"
  # sbatch train.sh
fi