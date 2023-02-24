python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder/biencoder_wiki_large.bin \
--data_path data/zeshel/blink_format --output_path models/zeshel/top --encode_batch_size 128 --eval_batch_size 1 \
--top_k 64 --save_topk_result --bert_model bert-large-uncased --mode valid --zeshel True --data_parallel \
--cand_encode_path data/zeshel/cand_enc/cand_enc_valid.pt --cand_pool_path data/zeshel/cand_pool/cand_pool_valid.pt >valid2.out

python blink/biencoder/eval_biencoder.py --path_to_model /home/jongsong/BLINK/models/zeshel/biencoder/biencoder_wiki_large.bin \
--data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 128 --eval_batch_size 1 \
--top_k 64 --save_topk_result --bert_model bert-large-uncased --mode test --zeshel True --data_parallel \
--cand_encode_path data/zeshel/cand_enc/cand_enc_test.pt --cand_pool_path data/zeshel/cand_pool/cand_pool_test.pt >test2.out



