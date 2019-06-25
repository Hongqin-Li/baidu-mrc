import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from model.BiDAF_rep import BiDAF as Model
from utils import DataProvider


checkpoint_path = './checkpoints/checkpoint.pt'

# Hyperparameters
embedding_dim = 768
hidden_size = 256

learning_rate = 0.001

batch_size = 10
save_per_steps = 10
num_epochs = 10

def train(epochs):

    provider = DataProvider()

    model = Model(D_emb=embedding_dim, D_H=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    e = 0
    cnt = 0

    # Check previously saved model
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e = checkpoint['epoch']
        model.train()
        print ('Load previous model and optimizer!')
    except:
        print ('No saved model found!')


    while e < epochs:

        print (f'Epoch: {e}')

        for docs, quests, begin_idxs, end_idxs, in provider.train_batch(batch_size=batch_size):

            model.zero_grad()

            begin_idxs_out, end_idxs_out = model(docs, quests) 

            loss = criterion(begin_idxs_out, begin_idxs) + criterion(end_idxs_out, end_idxs)
            loss.backward()
            optimizer.step()

            print (f'Loss: {loss}')

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


