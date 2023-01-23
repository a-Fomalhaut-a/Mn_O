import torch
from torch import nn
from config import model_opt
from node_model.full_connect_res import full_connect_res


class Node_creat(nn.Module):
    def __init__(self):
        super(Node_creat, self).__init__()
        which_model = model_opt['which_model']
        if which_model == 'full_connect_res':
            self.netN = full_connect_res()
        else:
            self.netN = full_connect_res()

    def forward(self, x):
        out = self.netN(x)
        return out

