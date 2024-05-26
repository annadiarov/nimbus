"""
Code for Cross-Attention Block extracted from the following repository:
    https://github.com/lucidrains/tf-bind-transformer
"""
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from nimbus.utils import default, exists


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            context_dim=None,
            dropout=0.
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x,
            context,
            mask=None,
            context_mask=None
    ):
        h = self.heads

        x = self.norm(x)
        context = self.context_norm(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(context_mask):
            mask_value = -torch.finfo(sim.dtype).max
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
