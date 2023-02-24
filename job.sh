python blink/crossencoder/train_cross.py \
  --data_path  models/zeshel/top64_candidates/ \
  --learning_rate 1e-05 --num_train_epochs 300 --max_context_length 128 --max_cand_length 128 \
  --train_batch_size 2 --eval_batch_size 2 \
  --type_optimization all_encoder_layers --data_parallel \
  --zeshel True --output_path models/zeshel/crossencoder/4layers_dim1024_lr-5 --image_path 4layers_dim1024_lr-5\
  --patience 10 --layers 4 

python blink/crossencoder/train_cross.py \
  --data_path  models/zeshel/top64_candidates/ \
  --learning_rate 1e-05 --num_train_epochs 300 --max_context_length 128 --max_cand_length 128 \
  --train_batch_size 2 --eval_batch_size 2 \
  --type_optimization all_encoder_layers --data_parallel \
  --zeshel True --output_path models/zeshel/crossencoder/6layers_dim1024_lr-5 --image_path 6layers_dim1024_lr-5\
  --patience 10 --layers 6
  
python blink/crossencoder/train_cross.py \
  --data_path  models/zeshel/top64_candidates/ \
  --learning_rate 1e-03 --num_train_epochs 300 --max_context_length 128 --max_cand_length 128 \
  --train_batch_size 2 --eval_batch_size 2 \
  --type_optimization all_encoder_layers --data_parallel \
  --zeshel True --output_path models/zeshel/crossencoder/6layers_dim1024_lr-3 --image_path 6layers_dim1024_lr-3\
  --patience 10 --layers 6
  