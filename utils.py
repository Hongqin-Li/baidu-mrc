import pandas as pd
import torch
import json
import random


# Since we cannot run bert-service in our laptop...and all the input are fake!
# When running in server, just modified this to True
has_server = True

# Modify path to preprocessed raw train file provided by baidu
train_file = '../DuReader/data/preprocessed/trainset/search.train.json'
test_file = '' # TODO
dev_file = '' # TODO



cnt = 0


bc = None
if has_server:
    from bert_serving.client import BertClient
    bc = BertClient(ip='localhost')


def str_to_tensor(s):
    # str: string e.g. a question string or a document string
    # return: tensor e.g. use bert-service to return a 768 dimensional tensor

    if bc is None:
        return torch.stack([torch.Tensor(768) for c in s]) # For test
    else: 
        # FIXME
        return torch.Tensor(bc.encode(['*' if c.isspace() else c  for c in s]))


def segmented_paras_to_str(sps):
    return '\n'.join([ ''.join(sp) for sp in sps])

# i.e. one sample input
def line_to_input(line):

    data = json.loads(line)

    id = data['question_id']
    question = data['question'] # string

    if len(data['fake_answers']) == 0:
        # print (f'id: {id} No answer provided')
        return None
    elif len(data['fake_answers']) > 1:
        print (f'id: {id} Warning: More than 1 anwers are provided.')
    
    answer = data['fake_answers'][0] # array: (number_of_answer) e.g. [ans1, ans2, ..., ansi] each answer is a string

    relevant_doc = segmented_paras_to_str(data['documents'][data['answer_docs'][0]]['segmented_paragraphs'])

    ans_start_idx = relevant_doc.find(answer)
    ans_end_idx = ans_start_idx + len(answer)

    if ans_start_idx == -1:
        print (f'id: {id} Cannot find provided answer')
        return None



    global cnt
    cnt += 1
    print (f'\ridx: {cnt} ', end='')

    return str_to_tensor(relevant_doc), str_to_tensor(question), torch.LongTensor([ans_start_idx]), torch.LongTensor([ans_end_idx])


def pad_input(docs, quests, bis, eis):
    # Pad and sort
    # TODO sort by docs length
    docs = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True)
    quests = torch.nn.utils.rnn.pad_sequence(quests, batch_first=True)
    return docs, quests, torch.LongTensor(bis), torch.LongTensor(eis)



def raw_json_to_input_batches(path, batch_size=100):

    docs, quests, begin_idxs, end_idxs = [], [], [], []

    cnt = 0

    with open(path, 'r') as f:
        
        for line in f:  

            inp = line_to_input(line)

            if inp == None: continue

            docs.append(inp[0]) # document tensor
            quests.append(inp[1]) # question tensor
            begin_idxs.append(inp[2]) # answer span tensor
            end_idxs.append(inp[3]) # answer span tensor

            cnt += 1
            if cnt == batch_size:
                
                yield pad_input(docs, quests, begin_idxs, end_idxs)

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

    


