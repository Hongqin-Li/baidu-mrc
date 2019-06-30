import pandas as pd
import torch
import json
import random

# from bert_serving.client import BertClient
# bc = BertClient(ip='localhost')

# Modify path to preprocessed raw train file provided by baidu
train_file = '../DuReader/data/preprocessed/trainset/search.train.json'
test_file = '../DuReader/data/preprocessed/testset/search.test.json'
dev_file = '../DuReader/data/preprocessed/devset/search.dev.json'
stopword_file = './data/stopwords'

char_pretrained_file = './data/char_pretrained'

yesorno_only = False

max_seq_len = 2000

cnt = 0
start_cnt = 0

stopwords = []
punctuations = {',', '.', '。', '!', '?', ':', ';', '(', ')'}


char_to_vec = None

previous_raw_docs = []

def print_count():
    global cnt
    cnt += 1
    print (f'\r {cnt} ', end='')

def get_stopwords():

    global stopwords

    with open(stopword_file, 'r') as f:
        a = f.read().split()
        stopwords = set(a)
        print ('Finish loading stopwords.')
        return
    print ('Warning: Load stopwords failed!')

get_stopwords()


def get_pretrained_char_embedding(path):

    
    char_to_vec = {}

    with open(path, 'r') as f:

        for line in f:
            word_and_vec = line.split()
            word = word_and_vec[0]
            vec = [float(s) for s in word_and_vec[1:]]
            char_to_vec[word] = vec 

        print ('Load pretrained embedding.')
        return char_to_vec

    print ('Get pretrained embedding failed!')

    return None


char_to_vec = get_pretrained_char_embedding(char_pretrained_file)


# TODO
# Only allow Chinese character and punctuations
def preprocess_str(s, get_map=False):
    res = ''
    ri = 0 # result's index

    idx_map = [0] * len(s) # used to reproduce original answer

    for i, c in enumerate(s):

        # if not c.isspace() and '\u4e00' <= c <= '\u9fa5' and c not in stopwords:

        if not c.isspace():
            if c in char_to_vec:
                res += c # equivalent to res[ri] = c
                idx_map[ri] = i
                ri += 1

    # print (res) # Test preprocessed string
    # input ()
    if get_map: return res, idx_map
    else: return res

def strs_to_tensors(strs):
    # strs: list of strings
    # return: tensor (number_of_strings, max_seq_len , 768)

    max_seq_len = max([len(s) for s in strs])

    ts = []
    for s in strs:
        t = torch.Tensor([char_to_vec[c] for c in s])
        ts.append(t)

    pts =  torch.nn.utils.rnn.pad_sequence(ts, batch_first=True)

    return pts

def segmented_paras_to_str(sps):
    
    return ''.join([ ''.join(sp) for sp in sps])

def parse_line_testset(line):

    data = json.loads(line)

    id = data['question_id']

    raw_docs = [ segmented_paras_to_str(doc['segmented_paragraphs'])  for doc in data['documents']  ]

    question = preprocess_str(data['question']) # string

    doc_idx_map_pairs = [preprocess_str(doc, get_map=True)  for doc in raw_docs]

    docs = [p[0] for p in doc_idx_map_pairs]
    idx_maps = [p[1] for p in doc_idx_map_pairs]

    return question, docs, raw_docs, idx_maps, data


def parse_line(line):
    # return string, string, int, int

    data = json.loads(line)

    id = data['question_id']

    question_type = data['question_type']

    if yesorno_only and question_type != 'YES_NO':
        return None

    if len(data['fake_answers']) == 0:
        # print (f'id: {id} No answer provided')
        return None
    elif len(data['fake_answers']) > 1:
        print (f'id: {id} Warning: More than 1 anwers are provided.')

    raw_doc = segmented_paras_to_str(data['documents'][data['answer_docs'][0]]['segmented_paragraphs'])

    question = data['question'] # string
    answer = data['fake_answers'][0] # string

    doc, idx_map = preprocess_str(raw_doc, get_map=True)
    answer = preprocess_str(answer)
    question = preprocess_str(question)

    ans_begin_idx = doc.find(answer)
    ans_end_idx = ans_begin_idx + len(answer) - 1

    if ans_begin_idx == -1 or ans_end_idx < ans_begin_idx:
        print (f'id: {id} Cannot find given answer in document.')
        return None

    
    return doc, question, ans_begin_idx, ans_end_idx, raw_doc, idx_map, id

def raw_json_to_input_batches(path, batch_size=100):

    docs, quests, begin_idxs, end_idxs = [], [], [], []

    raw_docs, idx_maps, ids = [], [], []
    raw_datas = []

    cnt = 0

    omit_cnt = 0

    with open(path, 'r') as f:
        
        for line in f:  

            if omit_cnt < start_cnt: 

                omit_cnt += 1
                print_count()
                continue

            inp = parse_line(line)

            if inp == None: continue

            if inp[0] == '' or inp[1] == '': continue


            docs.append(inp[0]) # document string
            quests.append(inp[1]) # question string

            begin_idxs.append(inp[2]) # answer span start index
            end_idxs.append(inp[3]) # answer span end index

            raw_docs.append(inp[4])
            idx_maps.append(inp[5])
            ids.append(inp[6])
            raw_datas.append(json.loads(line))

            print_count()

            cnt += 1
            if cnt == batch_size:
                
                yield strs_to_tensors(docs), strs_to_tensors(quests), torch.LongTensor(begin_idxs), torch.LongTensor(end_idxs), raw_docs, idx_maps, raw_datas

                # reset to collect next batch
                docs, quests, begin_idxs, end_idxs, raw_docs, idx_maps, ids, raw_datas = [], [], [], [], [], [], [], []
                cnt = 0

class DataProvider:

    def __init__(self):
        pass


    def train_batch(self, batch_size=1000):

        global cnt 
        cnt = 0

        for docs, quests, begin_idxs, end_idxs, _, _, _ in raw_json_to_input_batches(train_file, batch_size=batch_size):
            yield docs, quests, begin_idxs, end_idxs
            

    def dev_batch(self, batch_size=1000, get_raw=False):

        global cnt
        pre_cnt = cnt 
        cnt = 0

        for docs, quests, begin_idxs, end_idxs, raw_docs, idx_maps, raw_datas in raw_json_to_input_batches(dev_file, batch_size=batch_size):
            if get_raw:
                yield docs, quests, begin_idxs, end_idxs, raw_docs, idx_maps, raw_datas
            else: 
                yield docs, quests, begin_idxs, end_idxs

        cnt = pre_cnt

    def test_batch(self, get_raw=True):

        global cnt
        pre_cnt = cnt 
        cnt = 0

        with open(test_file, 'r') as f:

            for line in f:

                quest, docs, raw_docs, idx_maps, raw_data = parse_line_testset(line)
                print (f'question: {quest}')

                yield strs_to_tensors([quest] * len(docs)), strs_to_tensors(docs), raw_docs, idx_maps, raw_data
        
        cnt = pre_cnt
        

if __name__ == '__main__':
    
    # Usage
    dataset = DataProvider()


    # Notice that we have several candidate documents for one question, and thus the quests bellow is all same
    for quests, docs, raw_docs, idx_maps, raw_data in dataset.test_batch():
        print (raw_docs)
        print (f'quests: {quests.shape}, docs: {docs.shape}')
        input ()
        

    for docs, quests, begin_idxs, end_idxs, raw_docs, idx_maps, raw_datas in dataset.dev_batch(batch_size=3, get_raw=True):
        # get_raw: get raw_docs and idx_maps
        # This is necessary when we want to get back the original answer span from the preprocessed one

        print (docs.shape, quests.shape, begin_idxs.shape, end_idxs.shape)

        for d, bi, ei, idx_map in zip(raw_docs, begin_idxs, end_idxs, idx_maps):
            raw_ans = d[idx_map[bi]: idx_map[ei]]
            # This is the way to get back original answer
            print (raw_ans)
            
        input ()

    for docs, quests, begin_idxs, end_idxs in dataset.train_batch(batch_size=3):
        print (docs.shape, quests.shape, begin_idxs.shape, end_idxs.shape)
        input ()

    


