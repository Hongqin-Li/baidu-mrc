## Train

In our work, we choose a batch size of 10, since the embedding vector is 768 dim and the document length is about 1000~2000 in average. Thus the model cannot fit well into the GPU if larger batch size is applied.

The input of model $X$ is of shape (batch_size, seq_len, embedding_size), e.g. $X[i][j]$ is a vector of j-th word of i-th document 

And the output of the model are spans information with begin index $\hat{bi}​$ and end index $\hat {ei}​$, both of which are of shape (batch_size, seq_len), e.g. $\hat {bi}[i][j]​$ is the probability that begin index of i-th document occurs at j-th word of the i-th document.

As for the loss function, since both begin index and end index can be regarded as a classification problem of $seq\_len$ classes, we simply add up the cross entropy of both, i.e. $Loss= CrossEntropy(bi, \hat {bi}) + CrossEntropy(ei, \hat{ei})$, where $bi$, $ei$ are the target begin index and end index respectively and $\hat{bi}$, $\hat{ei}$ are the prediction of begin index and end index given by our model.



## Result

After training on the Baidu Search dataset for only **one** epoch, we generate the prediction on development set of Baidu Search and use the official scoring script to evaluation our result. The comparison with official BiDAF baseline is shown below. (Notice that the result given by below don't include the bonus for answering yes-or-no or entity, since we haven't implement this part. And that's why the scores below are much more less than those on the leader board. )

BiDAF Baseline provided by Baidu (From their [paper](https://arxiv.org/pdf/1711.05073.pdf))

| BLEU | Rouge-L |
| ---- | ------- |
| 23.1 | 31.1    |

Result with single model

| BLEU  | Rough-L |
| ----- | ------- |
| 22.56 | 32.53   |

Result with question-type-specific model, i.e., we train a model by only yes-or-no questions and another model by other questions, then the prediction of yes-or-no questions are given by the yes-or-no model and other questions are given by another model. (Because the answers for yes-or-no questions are always very short, which is quite different from description and entity questions)

| BLEU  | Rough-L |
| ----- | ------- |
| 20.35 | 33.04   |



### Summary

As we can see above, our model is able to achieve almost the same score of the baseline on this Chinese QA task. However, a quite strange thing is that although we only use char-level embedding and haven't included the word-level embedding, we can achieve almost the same score of baseline, which utilize both char and word level embedding. We think the possible reasons includes:

1. Chinese (in fact, ancient Chinese) character is exactly a word and maybe char-level embedding is quite suitable for Chinese NLP task
2. The high performance of pretrained vectors given by BERT.
3. Our preprocessing, only allowing 6000 Chinese character and punctuations, greatly removes the noise of the documents.