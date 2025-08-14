
import torch
from torch import nn
from torch_geometric_temporal import A3TGCN2, DCRNN, A3TGCN, AGCRN
import torch.nn.functional as F


class recurrent_DCRNN(torch.nn.Module):
    def __init__(self, node_features,dim_h1=256,out_features=128):
        super(recurrent_DCRNN, self).__init__()

        self.fc0 = nn.Linear(node_features, dim_h1)

        self.recurrent = DCRNN(dim_h1, out_features, 2)
        self.fc = nn.Linear(out_features, 1)

        self.e = torch.empty(3000, 4).cuda()

        torch.nn.init.xavier_uniform_(self.e)

        self.h = None
    def reshape(self,x):
        padding = torch.zeros(3000 - x.size(0), x.size(1)).cuda()
        x_padded = torch.cat((x, padding), dim=0)
        return x_padded
    def unreshape(self,x):
        padding = torch.zeros(3000 - x.size(0), x.size(1)).cuda()
        x_padded = torch.cat((x, padding), dim=0)
        return x_padded

    def forward(self, x, edge_index,edge_weight=None):
        e = self.e.detach()
        size = x.size()


        x = F.relu(self.fc0(x))

        h = self.recurrent(x, edge_index, edge_weight)
        out = self.fc(F.relu(h))

        torch.cuda.memory_allocated()
        out =  out.squeeze()

        return out, h
