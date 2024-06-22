"""
Code for self-attention block extracted from the following repository:
    https://github.com/lucidrains/tf-bind-transformer
"""

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from nimbus.utils import exists
from nimbus.ml_blocks import FeedForwardBlock


class SelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x,
            mask=None,
            return_attn=False,
    ):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if return_attn:
            return self.to_out(out), attn

        return self.to_out(out)


class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dropout=0.,
            ff_mult=4,
            **kwargs
    ):
        super().__init__()
        self.attn = SelfAttention(dim=dim, dropout=dropout, **kwargs)
        self.ff = FeedForwardBlock(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(self, x, mask=None, return_attn=False):
        if return_attn:
            x1, attn = self.attn(x, mask=mask, return_attn=return_attn)
            x = x1 + x
            x = self.ff(x) + x
            return x, attn
        else:
            x = self.attn(x, mask=mask, return_attn=False) + x
            x = self.ff(x) + x
            return x
