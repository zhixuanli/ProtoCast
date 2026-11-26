"""Patch utilities and gated attention blocks."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def patchify(x, p):
    b, t, c, h, w = x.shape
    return x.reshape(b, t, c, h // p, p, w // p, p).permute(
        0, 1, 3, 5, 2, 4, 6).reshape(b, t, h // p * (w // p), c * p * p)


def unpatchify(x, p, h, w):
    b, t, _, q = x.shape
    return x.reshape(b, t, h // p, w // p, q // (p * p), p, p).permute(
        0, 1, 4, 2, 5, 3, 6).reshape(b, t, q // (p * p), h, w)


def sinusoidal(n, d):
    position = torch.arange(n).float()[:, None]
    frequency = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
    embedding = torch.zeros(n, d)
    embedding[:, 0::2] = torch.sin(position * frequency)
    embedding[:, 1::2] = torch.cos(position * frequency)[:, :d // 2]
    return embedding


class GatedBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.gate_value = nn.Linear(dim, 4 * dim)
        self.output = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.dropout(self.attention(z, z, z, need_weights=False)[0])
        gate, value = self.gate_value(self.norm2(x)).chunk(2, -1)
        return x + self.dropout(self.output(F.silu(gate) * value))


