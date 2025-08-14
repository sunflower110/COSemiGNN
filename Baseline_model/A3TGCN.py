import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal import A3TGCN


class recurrent_A3TGCN(torch.nn.Module):
    def __init__(self, node_features,dim_h1=256,out_features=128, periods=2):
        super(recurrent_A3TGCN, self).__init__()

        self.fc0 = nn.Linear(node_features, dim_h1)

        self.recurrent = A3TGCN(dim_h1, out_features, periods)

        self.fc = nn.Linear(out_features, 1)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.fc0(x))
        x = x.view(x.shape[0], x.shape[1],1)

        h = self.recurrent(x, edge_index, edge_weight)

        out = self.fc(F.relu(h))

        out = out.squeeze()

        return out, h