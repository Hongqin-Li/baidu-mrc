import json
import os
import nltk
import torch

class Data:
    def __init__(self,c,q,ans,s_idx,e_idx):
        self.c=c
        self.q=q
        self.ans=ans
        self.s_id=s_idx
        self.e_id=e_idx

class DataSet:
    def __init__(self,path):
        self.data=[]
        with open(path, 'r', encoding='utf-8') as f:
            data_load = json.load(f)
            data_load = data_load['data']
            for article in data_load:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']

                    # TODO Here I need to tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)
                            l = 0
                            # s_found = False
                            # for i, t in enumerate(tokens):
                            #     while l < len(context):
                            #         if context[l] in abnormals:
                            #             l += 1
                            #         else:
                            #             break
                            #     # exceptional cases
                            #     if t[0] == '"' and context[l:l + 2] == '\'\'':
                            #         t = '\'\'' + t[1:]
                            #     elif t == '"' and context[l:l + 2] == '\'\'':
                            #         t = '\'\''
                            #     l += len(t)
                            #     if l > s_idx and s_found == False:
                            #         s_idx = i
                            #         s_found = True
                            #     if l >= e_idx:
                            #         e_idx = i
                            #         break
                            data_piece=Data(context,question,answer,s_idx,e_idx)
                            self.data.append(data_piece)
    
    # Get the 
    def get_char(self,b_id,e_id):
        c_char=[]
        q_char=[]

        for i in range(b_id,e_id):
            c_char.append(self.data[i].c)
            q_char.append(self.data[i].q)
        return c_char,q_char
    
    def get_targ(self,b_id,e_id):
        # TODO 
        # Input the beginning of the batch, and the end of the batch
        # Ouput the tensor describe the beginning  and the end of the answer
        return s_idx,e_idx 
    
    
        


    
