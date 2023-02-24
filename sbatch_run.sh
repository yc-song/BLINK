#!/bin/bash
#SBATCH --job-name=ep984exv
#SBATCH --output=slurm_output/%A-%a.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=128000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
# add sweep parameters
lr=`grep -o '"learning_rate": [^,]*' /home/jongsong/BLINK/models/zeshel/crossencoder/1ddpn1mo/training_params.json | grep -o '[^ ]*$'`
dim_red=`grep -o '"dim_red": [^,]*' /home/jongsong/BLINK/models/zeshel/crossencoder/1ddpn1mo/training_params.json | grep -o '[^ ]*$'`
layers=`grep -o '"layers": [^,]*' /home/jongsong/BLINK/models/zeshel/crossencoder/1ddpn1mo/training_params.json | grep -o '[^ ]*$'`
train_batch_size=`grep -o '"train_batch_size": [^,]*' /home/jongsong/BLINK/models/zeshel/crossencoder/1ddpn1mo/training_params.json | grep -o '[^ ]*$'`
decoding=`grep -o '"decoding": [^,]*' /home/jongsong/BLINK/models/zeshel/crossencoder/1ddpn1mo/training_params.json | grep -o '[^ ]*$'`
echo "lr=${lr}, dim_red=${dim_red}, layers=${layers}, train_batch_size=${train_batch_size}"
python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --decoding={$decoding} --run_id="1ddpn1mo" --dim_red=${dim_red} --layers=${layers} --learning_rate=${lr} --train_batch_size=${train_batch_size} --resume=True