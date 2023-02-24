#!/bin/sh

output=`python blink/crossencoder/train_cross.py --learning_rate $1 --num_train_epochs 1000 --train_batch_size 128 --eval_batch_size 128 --wandb "BERT with Speical Tokens" --save True --train_size 10000 --valid_size 10000 --architecture special_token --add_linear True`
run_id=${output: (-11):8}
val_acc=${output: (-2):2}
echo "$output"
echo "$run_id"
echo "$val_acc"
if  ((10#$val_acc!=0))
then
  sbatch sbatch_bert.sh $1 $run_id
fi