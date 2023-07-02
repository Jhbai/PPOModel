import torch
import torch.nn as nn

class actor(nn.Module):
    def __init__(self, state_dim, num_of_act): # state_dim為當下環境狀況, num_of_act為action的數量
        super(actor, self).__init__()

        '''這邊可以重塑模型'''
        self.Layer1 = nn.Linear(state_dim, 150)
        self.Layer2 = nn.Linear(150, 120)
        self.Layer3 = nn.Linear(120, num_of_act)

    def forward(self, x):

        '''這邊可以重塑模型'''
        X = nn.ReLU()(self.Layer1(x))
        X = nn.ReLU()(self.Layer2(X))
        X = nn.Softmax()(self.Layer3(X))
        return X
