import torch.nn.functional as F
from torch import nn


class MA(nn.Module):
    def __init__(self, feature_in=128, embed_mid=128, embed_out=256, num_heads=4):
        super(MA, self).__init__()

        self.embedding_position1 = nn.Linear(feature_in, embed_mid)
        self.embedding_position2 = nn.Linear(embed_mid, embed_out)

        self.linear_q = nn.Linear(embed_out, embed_out)
        self.linear_k = nn.Linear(embed_out, embed_out)
        self.linear_v = nn.Linear(embed_out, embed_out)

        self.attention = nn.MultiheadAttention(embed_dim=embed_out, num_heads=num_heads, batch_first=True)

        self.norm_trans_tail = nn.LayerNorm(embed_out, eps=1e-12)

    def forward(self, x):
        p = F.leaky_relu(self.embedding_position1(x))
        p = F.leaky_relu(self.embedding_position2(p))

        q = self.linear_q(p)
        k = self.linear_k(p)
        v = self.linear_v(p)

        # MultiheadAttention expects (batch, seq, embed_dim) if batch_first=True
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attention_output, _ = self.attention(q, k, v)
        p = attention_output.squeeze(1)

        p = self.norm_trans_tail(p)
        return p
