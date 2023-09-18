import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, kdim=256, vdim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert kdim % num_heads == 0
        assert vdim % num_heads == 0
        self.kdim = kdim
        self.vdim = vdim

        self.w_q = nn.Linear(embed_dim, kdim)
        self.w_k = nn.Linear(embed_dim, kdim)
        self.w_v = nn.Linear(embed_dim, vdim)
        self.w_o = nn.Linear(vdim, embed_dim)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        qk = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.kdim // self.num_heads))     # (N, num_heads, L, L)
        if mask is not None:
            qk = qk.masked_fill(mask.to(qk.dtype) == 0, float('-inf'))

        weights = F.softmax(qk, dim=-1)         # (N, num_heads, L, L)
        attention = torch.matmul(weights, v)    # (N, num_heads, L, vdim // num_heads)

        return attention, weights

    def split_heads(self, x):
        batch_size, seq_len = x.shape[:2]
        return x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        # qkv.shape: (N, L, E)
        batch_size, seq_len, _ = q.shape

        q = self.w_q(q)     # (N, L, kdim)
        k = self.w_k(k)     # (N, L, kdim)
        v = self.w_v(v)     # (N, L, vdim)

        q = self.split_heads(q)     # (N, num_heads, L, kdim // num_heads)
        k = self.split_heads(k)     # (N, num_heads, L, kdim // num_heads)
        v = self.split_heads(v)     # (N, num_heads, L, vdim // num_heads)

        attention, weights = self.scaled_dot_product_attention(q, k, v, mask)
        concat = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (N, L, vdim)
        output = self.w_o(concat)   # (N, L, E)

        return output, weights


if __name__ == '__main__':
    batch_size = 4
    seq_len = 20
    embed_dim = 64

    ma = MultiHeadAttention(embed_dim=embed_dim)
    x = torch.rand(batch_size, seq_len, embed_dim)
    y = ma(q=x, k=x, v=x)
    print(x.shape)
    print(y[0].shape, y[1].shape)

    print("\nDone")
