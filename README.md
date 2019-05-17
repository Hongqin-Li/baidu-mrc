# 百度机器阅读理解大赛

Structure:

```
.
├── materials: Contest descriptions
├── model: Baseline model
├── raw_data: ignored by .gitignore
├── data: ignored by .gitignore
├── utils: Data Preprocessing
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



## Task

Official description can be found [here](https://ai.baidu.com/broad/introduction?dataset=dureader).

### Description

Given a question $q$, and a set of documents $D = \{d_1, d_2, ..., d_n\}$, we are expected to give **an** answer $a$ as close as possible to reference answers $Ar = \{ar_1, ar_2, ..., ar_m\}$, according to the evidences in document $$D$$.



### Input





### Submission





### Evaluation

It will have a set of evaluation metrics (Bleu-4, Rouge-L etc.) to measure the **closeness** of our answer $a$ to the reference answer $Ar$.





## Optimization

**把答案加入encode**：由于description类的问题的某些备选答案是携带语义信息的，故将备选答案也做encoding处理。

