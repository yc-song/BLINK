#!/bin/bash

#SBATCH --job-name=blink
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=128000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1

iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
jobid=$(sbatch --train_sweep.sh)
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency train_sweep.sh)
    dependency=",${dependency}afterany:${jobid}"
done