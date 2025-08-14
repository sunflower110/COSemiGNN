import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal import LRGCN

class recurrent_LRGCN(torch.nn.Module):
    def __init__(self, node_features,dim_h1=256,out_features=128):
        super(recurrent_LRGCN, self).__init__()

        self.fc0 = nn.Linear(node_features, dim_h1)

        self.recurrent = LRGCN(dim_h1, out_features, 3, 3)
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
        num_edges = edge_index.shape[1]

        edge_weight = torch.ones((num_edges), dtype=torch.long)

        x = self.reshape(x)

        x = F.relu(self.fc0(x))

        h_0, c_0 = self.recurrent(x, edge_index,edge_weight, H=self.h_0, C=self.c_0)

        out = self.fc(h_0)
        self.h_0, self.c_0 = h_0.detach(), c_0.detach()

        out = out[:size[0]]
        out = out.squeeze()

        return out, h_0