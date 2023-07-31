#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_output/som/%j.out
#SBATCH --mem=200000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1

python blink/crossencoder/train_cross.py --resume=True --run_id=228eorg9 --architecture=mlp_with_som --decoder=False --dim_red=1536 --layers=3 --learning_rate=0.00171694327148553 --train_batch_size=64 --weight_decay=0.01 --dot_product True --eval_batch_size=128

# python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --architecture=mlp_with_som --decoder=True --dim_red=1536 --layers=2 --learning_rate=0.00011235861209414792 --num_train_epochs=50 --train_batch_size=8 --resume=True --run_id=9xh6r1v8 --wandb=mlp-with-som
# if [ "$time_out" = "time_out" ]; then
#   echo "timeout"
#   sbatch resume_raw_bert.sh $1
# else
#   echo "Python script ran successfully"
#   # sbatch train_raw_bert.sh
# fi