# 百度机器阅读理解大赛

Structure:

```
.
├── materials: Contest descriptions
├── model: BiDAF models
├── raw_data: ignored by .gitignore
├── data: ignored by .gitignore
├── utils: Data Preprocessing
│
├── train.py
├── test.py: Generate formatted pred.json and ref.json, which can be evaluated by `bash eval.sh`
├── submit.py: Generate submission file
│ 
├── evaluation_metric: official evaluation script
├── eval.sh: script for evaluation
│ 
└── README.md
```



## Push and Pull

Details can be found [here](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/)

First clone the repo by

```
git clone ...
```

- Never commit to the master directly.
- Always do your work on a different branch from master.

#### Basic Shared Repository Workflow

- update your local repo with `git pull origin master`,
- create a working branch with `git checkout -b MyNewBranch`
- make your changes on your branch and stage them with `git add`,
- commit your changes locally with `git commit -m "description of your commit"`, and
- upload the changes (including your new branch) to GitHub with `git push origin MyNewBranch`
- Go to the main repo on GitHub where you should now see your new branch
- click on your branch name
- click on “Pull Request” button (URC)
- click on “Send Pull Request”



## Usage

First, run bert-service since and create a pretrained word embedding file.

```shell
# Under ./baidu-mrc
cd bert-serving
bash run_server.sh
# ...wait for a while
cd ../data
python3 create_dict.py
```

which will generate some temp file in `./bert-serving` and a Chinese character embedding file at `data/char_pretrained`.

Then train by 

```shell
# Under ./baidu-mrc
python3 train.py
```

The saved model locates at  `./checkpoints/checkpoint.pt`

(Optionally) train a yes-or-no model by modifing `yesorno_only = True` in  `utils.py` and then running

```
python3 train.py
```



### Test

(Optionally) Turn `use_yesorno_model = True` in `test.py` if you have trained a yes-or-no model and want to use it to create prediction.

Test on development dataset by 

```
python3 test.py
```

which will generate `../pred.json` and `../ref.json` for prediction and reference respectively. Then evaluate our result by 

```
bash eval.sh
```



### Submit

(Optionally) Turn `use_yesorno_model = True` in `submit.py` if you have trained a yes-or-no model and want to use it to create prediction.

Then run the following command to generate a submission file `./submit.json`.

```
python3 submit.py
```





## Task

Official description can be found [here](https://ai.baidu.com/broad/introduction?dataset=dureader).

### Description

Given a question $q$, and a set of documents $D = \{d_1, d_2, ..., d_n\}$, we are expected to give **an** answer $a$ as close as possible to reference answers $Ar = \{ar_1, ar_2, ..., ar_m\}$, according to the evidences in document $$D$$.



### Submission

[link](https://ai.baidu.com/broad/submission?dataset=dureader)



### Evaluation

It will have a set of evaluation metrics (Bleu-4, Rouge-L etc.) to measure the **closeness** of our answer $a$ to the reference answer $Ar$.


