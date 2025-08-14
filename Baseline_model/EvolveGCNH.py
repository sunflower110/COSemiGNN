import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal import EvolveGCNH



class recurrent_EvolveGCNH(nn.Module):
    def __init__(self, node_features, hidden_dim=256, output_dim=1):
        super(recurrent_EvolveGCNH, self).__init__()

        self.fc0 = nn.Linear(node_features, hidden_dim)


        self.recurrent = EvolveGCNH(num_of_nodes=3000,in_channels=hidden_dim)


        self.fc = nn.Linear(hidden_dim, output_dim)

        self.h_0 = None

    def reshape(self, x):

        if x.size(0) < 3000:
            padding = torch.zeros(3000 - x.size(0), x.size(1)).to(x.device)
            x = torch.cat((x, padding), dim=0)
        return x

    def forward(self, x, edge_index, edge_weight=None):
        size = x.size()
        x = self.reshape(x)
        x = F.relu(self.fc0(x))


        h = self.recurrent(x, edge_index, edge_weight)

        out = self.fc(F.relu(h))
        out = out[:size[0]]

        out = out.squeeze()

        return out, h.detach()
