


BERT_BASE_DIR=../bert_base/chinese_L-12_H-768_A-12

bert-serving-start -model_dir $BERT_BASE_DIR -num_worker=1 >& log
