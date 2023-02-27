## Example to train BLINK on Zero-shot Entity Linking dataset

### Conda environment setup. 
To run the code, create a new conda environment by running the following command:

    conda create -n blink -y python=3.7 && conda activate blink
    pip install -r requirements.txt
 
### Download dataset
Download the Zero-shot Entity Linking dataset by running the following command:

    ./examples/zeshel/get_zeshel_data.sh
 
### Converting data to BLINK format
To convert the downloaded data into BLINK format, run:
    python examples/zeshel/create_BLINK_zeshel_data.py

NOTE: If you add '--debug' to the following commands, the debug mode is activated and test the code for the limited number (200) of datasets.

### Training Biencoder model.
To train the biencoder model, run: 

    python blink/biencoder/train_biencoder.py --optimizer AdamW --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 128 --eval_batch_size 64 --bert_model bert-base-cased --type_optimization all_encoder_layers --data_parallel True
Please change the batch size accordingly to your computing resources.

### Getting top-k predictions from Biencoder model
To get top-k predictions from the biencoder model on each dataset (train/valid/test), run:

    python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 256 --eval_batch_size 32 --top_k 64 --bert_model bert-base-cased --mode train,valid,test --zeshel True --data_parallel True --architecture special_tokens --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool --save_topk_result 

The context ([mention_start embedding) and top_k candidates ([entity] embeddings) are stored in the output_path with scores and labels.

### Training and Evaluating Crossencoder Model
To train and evaluate the crossencoder model, follow the below steps:

1. Login to wandb to track the experiment.
2. For Feed forward network architecture (Approach 1), run:
    python blink/crossencoder/train_cross.py --act_fn=softplus --architecture=mlp --decoder=False --dim_red=512 --layers=6 --learning_rate=0.001 --train_batch_size=256

3. For BERT whose inputs are [Mention_start] and [ENT] token embedding from bi-encoder (Approach 2), run:
    python blink/crossencoder/train_cross.py --wandb <your project name> --architecture special_token --learning_rate 2e-05 --num_train_epochs 100 --train_batch_size 256 --eval_batch_size 1024 --wandb "BERT with Speical Tokens" --save True --add_linear True

For crossencoder whose inputs are "raw context text"+"[ENT] embedding" (Approach 3), the retriever should save raw context text instead of [mention_start] embedding. Thus, you should run bi-encoder with different arguments.

1. Running bi-encoder and get candidates:
    python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder_base/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel_base_special_tokens --encode_batch_size 128 --eval_batch_size 128 --top_k --save_topk_result --bert_model bert-base-cased --mode valid --zeshel True --data_parallel True --architecture raw_context_text --cand_encode_path data/zeshel/cand_enc --cand_pool_path data/zeshel/cand_pool_base/cand_pool

2. Train and evaluate cross-encoder
    blink/crossencoder/train_cross.py --wandb <your project name> --architecture raw_context_text --learning_rate 2e-05 --num_train_epochs 20 --train_batch_size 8 --eval_batch_size 128 --save True --add_linear True