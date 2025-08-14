
import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
class GAT(torch.nn.Module):

    def __init__(self, input=182, dim_h=128, dim_out=64, heads=8):
        super(GAT, self).__init__()
        self.norm1 = BatchNorm1d(input)
        self.gcn1 = GCNConv(input, dim_h)
        self.norm2 = BatchNorm1d(dim_h*heads)
        self.gcn2 = GCNConv(dim_h*heads, dim_out)
        self.fc = nn.Linear(dim_out, 2)

    def forward(self,x,adj):
        h = self.norm1(x)
        h = self.gcn1(h, adj)
        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = self.gcn2(h, adj)
        out = self.fc(F.relu(h))

        out = out.squeeze()
        return out,h


