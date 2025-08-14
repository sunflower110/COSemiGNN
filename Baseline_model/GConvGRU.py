import torch
from torch import nn
from torch_geometric_temporal import GConvGRU
import torch.nn.functional as F

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, 128, 2)
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
class recurrent_GConvGRU(torch.nn.Module):
    def __init__(self, node_features,dim_h1=256,out_features=128):
        super(recurrent_GConvGRU, self).__init__()

        self.fc0 = nn.Linear(node_features, dim_h1)

        self.recurrent = GConvGRU(dim_h1, out_features, 2)
        self.fc = nn.Linear(out_features, 1)

        self.h_0,self.c_0 = None,None

    def reshape(self,x):
        padding = torch.zeros(3000 - x.size(0), x.size(1)).cuda()
        x_padded = torch.cat((x, padding), dim=0)
        return x_padded
    def unreshape(self,x):
        padding = torch.zeros(3000 - x.size(0), x.size(1)).cuda()
        x_padded = torch.cat((x, padding), dim=0)
        return x_padded

    def forward(self, x, edge_index, edge_weight=None):

        size = x.size()

        x = self.reshape(x)

        x = F.relu(self.fc0(x))

        h = self.recurrent(x, edge_index, edge_weight)

        out = self.fc(F.relu(h))

        out = out.squeeze()
        out = out[:size[0]]

        out = out.squeeze()
        return out, h