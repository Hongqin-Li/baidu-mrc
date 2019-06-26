


BERT_BASE_DIR=../../bert_base/chinese_L-12_H-768_A-12
MAX_SEQ_LEN=10

# bert-serving-start -pooling_strategy NONE -max_seq_len $MAX_SEQ_LEN -show_tokens_to_client -model_dir $BERT_BASE_DIR -num_worker=1 >& log
bert-serving-start -show_tokens_to_client -model_dir $BERT_BASE_DIR -num_worker=1 >& log
