
import torch
import os
import json

from utils import DataProvider
from train import get_model_and_optimizer
from train import checkpoint_path, yesorno_checkpoint_path

submit_file = './submit.json'
use_yesorno_model = False

def submit():

    if os.path.exists(submit_file):
        print ('Find old submit file. Want to create a new one? (y/n) ', end='')
        a = input ()
        if a == 'y':
            os.remove(submit_file)
            print ('Remove old submit file')
        elif a == 'n': 
            new_ref = False
        else:
            print ('Invalid input')
            return 
 

    data_provider = DataProvider()
    model, _ = get_model_and_optimizer()

    yesorno_model = None
    if use_yesorno_model:
        yesorno_model, _ = get_model_and_optimizer(yesorno_checkpoint_path)

    cnt = 0
    # TODO Actually test_batch batch size is 1
    for quests, docs, raw_docs, idx_maps, raw_data in data_provider.test_batch():
    
        if torch.cuda.is_available():
            quests = quests.cuda()
            docs = docs.cuda()

        begin_idxs_out, end_idxs_out = None, None
        if use_yesorno_model and raw_data['question_type'] == 'YES_NO':
            begin_idxs_out, end_idxs_out = yesorno_model(docs, quests) 
        else:
            begin_idxs_out, end_idxs_out = model(docs, quests) 
        
        begin_idxs = torch.argmax(begin_idxs_out, dim=1).tolist()
        end_idxs = torch.argmax(end_idxs_out, dim=1).tolist()

        answers = []
        for d, bi, ei, idx_map in zip(raw_docs, begin_idxs, end_idxs, idx_maps):
            raw_ans = d[idx_map[bi]: idx_map[ei] + 1]
            # print (f'doc: {d}')
            # print (f'answer: {raw_ans}')
            # input ()
            answers.append(raw_ans)

        # TODO select the best answer according to softmax
        # Construct json 
        pred = {}
        pred['yesno_answers'] = []
        pred['question'] = raw_data['question']
        pred['question_type'] = raw_data['question_type']
        pred['answers'] = answers
        pred['question_id'] = raw_data['question_id']
        pred_s = json.dumps(pred, ensure_ascii=False)

        with open(submit_file, 'a') as f:
            f.write(pred_s + '\n')

            cnt += 1
            print (f'\r {cnt} ', end='')


if __name__ == '__main__':
    submit()

