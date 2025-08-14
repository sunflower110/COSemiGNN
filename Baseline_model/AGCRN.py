
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal import AGCRN


class recurrent_AGCRN(torch.nn.Module):
    def __init__(self, node_features,dim_h1=128,out_features=32):
        super(recurrent_AGCRN, self).__init__()

        self.fc0 = nn.Linear(node_features, dim_h1)

        self.recurrent = AGCRN(number_of_nodes = 3000,
                              in_channels = dim_h1,
                              out_channels = out_features,
                              K = 2,
                              embedding_dimensions = 4)
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

    def forward(self, x, e):
        e = self.e
        size = x.size()

        x = self.reshape(x)
        x = x.view(1, x.shape[0], x.shape[1])
        x = F.relu(self.fc0(x))

        h_0 = self.recurrent(x, e, self.h)
        out = self.fc(F.relu(h_0))
        self.h = h_0.detach()
        out = out.squeeze()
        out = out[:size[0]]
        out = out.squeeze()

        torch.cuda.memory_allocated()

        return out, h_0
