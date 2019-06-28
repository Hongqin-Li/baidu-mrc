
import torch

from utils import DataProvider
from train import get_model_and_optimizer


def submit():

    data_provider = DataProvider()
    model, _ = get_model_and_optimizer()

    for quests, docs, raw_docs, idx_maps, id in data_provider.test_batch():
    
        if torch.cuda.is_available():
            quests = quests.cuda()
            docs = docs.cuda()

        begin_idxs_out, end_idxs_out = model(docs, quests) 
        
        begin_idxs = torch.argmax(begin_idxs_out, dim=1).tolist()
        end_idxs = torch.argmax(end_idxs_out, dim=1).tolist()

        for d, bi, ei, idx_map in zip(raw_docs, begin_idxs, end_idxs, idx_maps):
            raw_ans = d[idx_map[bi]: idx_map[ei] + 1]
            print (f'doc: {d}')
            print (f'answer: {raw_ans}')
            input ()
            
        # TODO select the best answer according to softmax


if __name__ == '__main__':
    submit()
