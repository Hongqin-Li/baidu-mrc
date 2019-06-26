import pandas as pd
import torch
import json
import random
from bert_serving.client import BertClient
bc = BertClient(ip='localhost')

# Modify path to preprocessed raw train file provided by baidu
train_file = '../DuReader/data/preprocessed/trainset/search.train.json'
test_file = '' # TODO
dev_file = '' # TODO

cnt = 0

# TODO
def preprocess_str(s):
    res = ''
    for c in s:
        if '\u4e00' <= c <= '\u9fa5':
            res += c
    return res

def strs_to_tensors(strs):
    # strs: list of strings
    # return: tensor (number_of_strings, max_seq_len , 768)

    max_seq_len = max([len(s) for s in strs])

    ts = []
    for s in strs:
        t = torch.Tensor(bc.encode(['$' if c.isspace() else c for c in s]))
        ts.append(t)

    pts =  torch.nn.utils.rnn.pad_sequence(ts, batch_first=True)

    return pts

def segmented_paras_to_str(sps):
    return ''.join([ ''.join(sp) for sp in sps])


def parse_line(line):
    # return string, string, int, int

    data = json.loads(line)

    id = data['question_id']

    if len(data['fake_answers']) == 0:
        # print (f'id: {id} No answer provided')
        return None
    elif len(data['fake_answers']) > 1:
        print (f'id: {id} Warning: More than 1 anwers are provided.')

    doc = segmented_paras_to_str(data['documents'][data['answer_docs'][0]]['segmented_paragraphs'])

    question = data['question'] # string
    answer = data['fake_answers'][0] # string

    doc = preprocess_str(doc)
    answer = preprocess_str(answer)
    question = preprocess_str(question)

    # print (f'doc_len: {len(doc)}, quest_len: {len(question)}, ans_len: {len(answer)}')

    ans_begin_idx = doc.find(answer)
    ans_end_idx = ans_begin_idx + len(answer)
    
    if ans_begin_idx == -1:
        print (f'id: {id} Cannot find given answer in document.')
        return None

    global cnt
    cnt += 1
    print (f'\ridx: {cnt} ', end='')
    
    return doc, question, ans_begin_idx, ans_end_idx

def raw_json_to_input_batches(path, batch_size=100):

    docs, quests, begin_idxs, end_idxs = [], [], [], []

    cnt = 0

    with open(path, 'r') as f:
        
        for line in f:  

            inp = parse_line(line)

            if inp == None: continue

            if inp[0] == '' or inp[1] == '': continue

            docs.append(inp[0]) # document string
            quests.append(inp[1]) # question string

            begin_idxs.append(inp[2]) # answer span start index
            end_idxs.append(inp[3]) # answer span end index

            cnt += 1
            if cnt == batch_size:
                
                yield strs_to_tensors(docs), strs_to_tensors(quests), torch.LongTensor(begin_idxs), torch.LongTensor(end_idxs)

                # reset to collect next batch
                docs, quests, begin_idxs, end_idxs = [], [], [], []
                cnt = 0


    return inputs, targets

class DataProvider:

    def __init__(self):
        pass


    def train_batch(self, batch_size=1000):

        global cnt 
        cnt = 0

        for batch in raw_json_to_input_batches(train_file, batch_size=batch_size):
            yield batch
            

if __name__ == '__main__':
    
    # Usage
    dataset = DataProvider()

    for docs, quests, begin_idxs, end_idxs in dataset.train_batch(batch_size=3):
        print (docs.shape, quests.shape, begin_idxs.shape, end_idxs.shape)
        input ()

    


