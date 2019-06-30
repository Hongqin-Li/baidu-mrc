import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from model.BiDAF_rep import BiDAF as Model
from utils import DataProvider
from utils import yesorno_only

# checkpoint_path = './checkpoints/checkpoint_yesorno.pt' if yesorno_only else './checkpoints/checkpoint.pt'

checkpoint_path = './checkpoints/checkpoint.pt'
yesorno_checkpoint_path = './checkpoints/checkpoint_yesorno.pt'

# Hyperparameters
embedding_dim = 768
hidden_size = 256

learning_rate = 0.001

batch_size = 10
save_per_steps = 10000//10
num_epochs = 10


def get_model_and_optimizer(checkpoint=checkpoint_path):

    model = Model(D_emb=embedding_dim, D_H=hidden_size)

    if torch.cuda.is_available():
        model = model.cuda()
        print ('Using GPU')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # e = checkpoint['epoch']
        model.train()
        print ('Load previous model and optimizer!')
    except:
        print ('No saved model found!')

    return model, optimizer



def test():
    provider = DataProvider()
    model, optimizer = get_model_and_optimizer()
    criterion = nn.CrossEntropyLoss()

    for docs, quests, begin_idxs, end_idxs in provider.dev_batch(batch_size=batch_size):

        if torch.cuda.is_available():
            docs = docs.cuda()
            quests = quests.cuda()
            begin_idxs = begin_idxs.cuda()
            end_idxs = end_idxs.cuda()

        model.zero_grad()


        begin_idxs_out, end_idxs_out = model(docs, quests) 

        # TODO How to calculate accuracy?
        begin_idxs_diff = torch.mean(torch.abs(torch.argmax(begin_idxs_out, dim=1) - begin_idxs).double())
        end_idxs_diff = torch.mean(torch.abs(torch.argmax(end_idxs_out, dim=1) - end_idxs).double())

        loss = criterion(begin_idxs_out, begin_idxs) + criterion(end_idxs_out, end_idxs)
        print (f'Loss: {loss}')
        print (f'begin/end idx diff: {begin_idxs_diff}, {end_idxs_diff}')
   

def train(epochs):

    e = 0
    cnt = 0

    provider = DataProvider()
    model, optimizer = get_model_and_optimizer()
    criterion = nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load(checkpoint_path)
        e = checkpoint['epoch']
    except:
        print ('No checkpoint found.')


    while e < epochs:

        print (f'Epoch: {e}')

        for docs, quests, begin_idxs, end_idxs, in provider.train_batch(batch_size=batch_size):

            if torch.cuda.is_available():
                docs = docs.cuda()
                quests = quests.cuda()
                begin_idxs = begin_idxs.cuda()
                end_idxs = end_idxs.cuda()

            model.zero_grad()

            print (f'docs len: {docs.shape[1]} ', end='')

            try:
                begin_idxs_out, end_idxs_out = model(docs, quests) 
                loss = criterion(begin_idxs_out, begin_idxs) + criterion(end_idxs_out, end_idxs)
                loss.backward()
                optimizer.step()
            except: 
                print ('Error when feed into model.')
                continue


            print (f'Loss: {loss}')
            # FIXME
            with open ('loss.log', 'a') as f:
                f.write(f'{loss}\n')

            cnt += 1
            if cnt == save_per_steps:
                cnt = 0
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print (f'Save model at epoch {e}' )

        e += 1


if __name__ == '__main__':

    train(num_epochs)


