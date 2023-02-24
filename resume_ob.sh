#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/th6hii2v/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/th6hii2v/%j.error
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate blink
PYTHONPATH=.

##mlp (64 candidates)
python /home/jongsong/BLINK/blink/crossencoder/train_cross.py --act_fn="sigmoid" --decoder=True --dim_red=768 --layers=6 --learning_rate=0.0007713085326136375 --train_batch_size=384 --resume=True --run_id=obvdx356
