import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')

class BiDAF(nn.Module):
    def __init__(self,D_emb,D_H,num_of_layers=2):
        super(BiDAF,self).__init__()
        # Context embedding
        # ( D_batch, sl, D_emb ) -> ( D_batch, sl, 2*D_H )
        self.LSTM_Context=nn.LSTM(
            input_size=D_emb,
            hidden_size=D_H,
            num_layers=num_of_layers,
            bidirectional=True
        )

        # Attention layer
        # (D_batch, 2*D_H, 1)
        self.AttC=nn.Linear(2*D_H,1)
        self.AttQ=nn.Linear(2*D_H,1)
        self.AttCQ=nn.Linear(2*D_H,1)

        # Modeling layer
        # ( D_batch, sl, 8*D_H) -> ( D_batch, sl, 2*D_H )        
        self.LSTM_Modeling1=nn.LSTM(
            input_size=8*D_H,
            hidden_size=D_H,
            num_layers=num_of_layers,
            bidirectional=True
        )
        
        # ( D_batch, sl, 2*D_H ) -> ( D_batch, sl, 2*D_H )
        self.LSTM_Modeling2=nn.LSTM(
            input_size=2*D_H,
            hidden_size=D_H,
            num_layers=num_of_layers,
            bidirectional=True
        )

        # Output layer
        # One more LSTM
        self.LSTMoutput=nn.LSTM(
            input_size=2*D_H,
            hidden_size=D_H,
            num_layers=num_of_layers,
            bidirectional=True
        )
        # (D_batch, sl, 8*D_H) -> (D_batch, sl, 1)
        self.p1g=nn.Linear(8*D_H,1)
        self.p1m=nn.Linear(2*D_H,1)
        
        self.p2g=nn.Linear(8*D_H,1)
        self.p2m=nn.Linear(2*D_H,1)
        # end

        self.softmax=nn.Softmax()
    
    def att_layer(self,c,q):
        D_batch=c.shape[0]
        sl_c=c.shape[1]
        sl_q=q.shape[1]
        S=torch.zeros(sl_q,sl_c,D_batch)
        # (D_batch, sl_c, 1)
        Ac=self.AttC(c).squeeze(dim=2).t()
        # (D_batch, 1, sl_q)
        Aq=self.AttQ(q).squeeze(dim=2).t()
        S+=Ac

        for i in range(sl_q):
            q_j=(q[:,i].t()*c.permute(1,2,0)).permute(2,0,1)
            # q_j ( D_batch , sl_c , 1 )
            q_j=self.AttCQ(q_j).permute(2,1,0).squeeze(dim=0)
            S[i]+=q_j

        S=S.permute(1,0,2)
        S+=Aq
        S=S.permute(2,0,1)
        # c2q
        # a (D_batch, sl_c, sl_q)
        a=F.softmax(S,dim=2)
        # c2q (D_batch, sl_c, 2*D_H)
        c2q=torch.bmm(a,q)

        # q2c
        # b (D_batch, 1, sl_q)
        b = F.softmax(torch.max(S, dim=2)[0], dim=1).unsqueeze(1)
        # q2c (D_batch, 1, 2*D_H)
        q2c=torch.bmm(b,c).squeeze()
        q2c=q2c.unsqueeze(1).expand(-1,sl_c,-1)

        return torch.cat((c,c2q,c*c2q,c*q2c),dim=2)


    def forward(self,context,quary):
        # context: (D_batch, sl, D_emb)
        # quary: (D_batch, sl_q, D_emb)
        c=self.LSTM_Context(context)[0]
        q=self.LSTM_Context(quary)[0]
        # c ( D_batch, sl, 8*D_H)
        # g ( D_batch, sl, 8*D_H)
        g=self.att_layer(c,q)
        x=self.LSTM_Modeling1(g)[0]
        m=self.LSTM_Modeling2(x)[0]
        p1=self.p1g(g)+self.p1m(m)
        
        m2=self.LSTMoutput(m)[0]
        p2=self.p2g(g)+self.p2m(m2)
        
        p1=F.softmax(p1.squeeze(dim=2),dim=1)
        p2=F.softmax(p2.squeeze(dim=2),dim=1)

        # print(m2)
        return p1,p2




