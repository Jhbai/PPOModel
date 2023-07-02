import torch
import torch.nn as nn

def SetUp(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_normal_(model.weight)
        model.bias.data.fill(0)