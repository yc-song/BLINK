#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-12:00:00
#SBATCH --mem=128000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error

iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
jobid=$(sbatch --parsable train_sweep_mlp_with_som.sh)
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency train_sweep_mlp_with_som.sh)
    dependency=",${dependency}afterany:${jobid}"
done