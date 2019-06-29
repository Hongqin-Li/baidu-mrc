
import torch
import os
import json

from utils import DataProvider
from train import get_model_and_optimizer



pred_file = '../pred.json'
ref_file = '../ref.json'

# Generate formatted prediction and reference file of development set
def test():
    
    new_ref = True
    
    if os.path.exists(ref_file):
        print ('Find old ref file. Want to create a new one? (y/n)', end='')
        a = input ()
        if a == 'y':
            os.remove(ref_file)
            print ('Remove old ref file')
        elif a == 'n': 
            new_ref = False
        else:
            print ('Invalid input')
            return 
            
       
    if os.path.exists(pred_file):
        os.remove(pred_file)
        print ('Remove old pred file.')

    data_provider = DataProvider()
    model, _ = get_model_and_optimizer()

    for docs, quests, begin_idxs, end_idxs, raw_docs, idx_maps, raw_datas in data_provider.dev_batch(batch_size=2, get_raw=True):
    
        if torch.cuda.is_available():
            quests = quests.cuda()
            docs = docs.cuda()

        # print (docs.shape, quests.shape)
        begin_idxs_out, end_idxs_out = model(docs, quests) 
        
        begin_idxs_pred = torch.argmax(begin_idxs_out, dim=1).tolist()
        end_idxs_pred = torch.argmax(end_idxs_out, dim=1).tolist()

        answer_preds = []
        for d, bi, ei, idx_map in zip(raw_docs, begin_idxs_pred, end_idxs_pred, idx_maps):
            raw_ans = d[idx_map[bi]: idx_map[ei] + 1]
            # print (f'doc: {d}')
            # print (f'answer: {raw_ans}')
            answer_preds.append(raw_ans)
            # input ()

        # TODO select the best answer according to softmax

        for answer, data in zip(answer_preds, raw_datas):
            # Construct pred.json 
            if answer == '': continue

            pred = {}
            pred['yesno_answers'] = []
            pred['question'] = data['question']
            pred['question_type'] = data['question_type']
            pred['answers'] = [answer]
            pred['question_id'] = data['question_id']
            pred_s = json.dumps(pred, ensure_ascii=False)

            with open(pred_file, 'a') as f:
                f.write(pred_s + '\n')

            # Construct ref.json
            if new_ref:
                ref = {}
                ref['yesno_answers'] = []
                ref['entity_answers'] = [[]]
                ref['source'] = 'search'
                ref['question'] = data['question']
                ref['question_type'] = data['question_type']
                ref['answers'] = data['answers']
                ref['question_id'] = data['question_id']
                ref_s = json.dumps(ref, ensure_ascii=False)

                with open(ref_file, 'a') as f:
                    f.write(ref_s + '\n')

if __name__ == '__main__':
    test()
