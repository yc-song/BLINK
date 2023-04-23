#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-12:00:00
#SBATCH --mem=128000MB
#SBATCH --cpus-per-task=8
#SBATCH --partition=P1
#SBATCH --output=/home/jongsong/BLINK/slurm_output/bert/%j.out
#SBATCH --error=/home/jongsong/BLINK/slurm_output/bert/%j.error

iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable train_mlp_with_bert.sh)
jobid=305091
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency eval_bi.sh)
    dependency=",${dependency}afterany:${jobid}"
done