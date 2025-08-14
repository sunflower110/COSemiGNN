import torch
from torch import nn
from torch_geometric_temporal import GCLSTM
import torch.nn.functional as F

class recurrent_GCLSTM(torch.nn.Module):
    def __init__(self, node_features,dim_h1=256,out_features=128):
        super(recurrent_GCLSTM, self).__init__()

        self.fc0 = nn.Linear(node_features, dim_h1)

        self.recurrent = GCLSTM(dim_h1, 128, 2)
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

        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, self.h_0, self.c_0)

        out = self.fc(F.relu(h_0))
        self.h_0, self.c_0 = h_0.detach(), c_0.detach()
        out = out.squeeze()
        out = out[:size[0]]
        out = out.squeeze()

        return out, h_0