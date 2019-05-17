# Data Preprocessing



## Preprocessed Raw Data

Raw:

```shell
wget -c https://aipedataset.cdn.bcebos.com/dureader/dureader_raw.zip
```



Preprocessed Raw data (This is what we need):

```bash
wget -c https://aipedataset.cdn.bcebos.com/dureader/dureader_preprocessed.zip
```

More can be found [here](https://github.com/baidu/DuReader/blob/master/data/download.sh)



## DuReader to SQuAD

Usage (e.g. convert 1000 samples):

```
python3 du_to_squad.py ../raw_data/preprocessed/trainset/zhidao.train.json --num_samples 1000
```





## Chunks

Usage:

```bash
bash split_and_format_raw.sh PATH_TO_RAW_JSON
```

