#!/bin/bash
#SBATCH --job-name=mlp-with-som-resume
#SBATCH --gres=gpu:3
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
# lr=`grep -o '"learning_rate": [^,]*' ./models/zeshel/crossencoder/mlp_with_som/$1/training_params/training_params.json | grep -o '[^ ]*$'`
# num_train_epochs=`grep -o '"num_train_epochs": [^,]*' ./models/zeshel/crossencoder/mlp_with_som/$1/training_params/training_params.json | grep -o '[^ ]*$'`

# echo "lr=${lr}"
python_output=$(python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --architecture=mlp_with_som --decoder=False --dim_red=768 --layers=4 --learning_rate=0.00046488765457263783 --num_train_epochs=40 --train_batch_size=8 --resume=True --run_id=$1 --wandb=mlp-with-som)
echo $python_output
time_out=$(echo "$python_output" | grep -E "time_out")
# if [ "$time_out" = "time_out" ]; then
#   echo "timeout"
#   sbatch resume_raw_bert.sh $1
# else
#   echo "Python script ran successfully"
#   # sbatch train_raw_bert.sh
# fi