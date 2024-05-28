"""
Code for Cross-Attention Block extracted from the following repository:
    https://github.com/lucidrains/tf-bind-transformer
"""
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from nimbus.utils import default, exists
from nimbus.ml_blocks import FeedForwardBlock, BidirectionalCrossAttentionBlock


class JointCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            context_dim=None,
            ff_mult=8,
            dropout=0.,
            **kwargs
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.attn = BidirectionalCrossAttentionBlock(dim=dim,
                                                    context_dim=context_dim,
                                                    dropout=dropout,
                                                    prenorm=True,
                                                    talking_heads=True,
                                                    **kwargs)

        self.ff = FeedForwardBlock(dim, mult=ff_mult, dropout=dropout)
        self.context_ff = FeedForwardBlock(context_dim, mult=ff_mult, dropout=dropout)

    def forward(
            self,
            x,
            context,
            mask=None,
            context_mask=None
    ):
        attn_out, context_attn_out = self.attn(x, context, mask=mask,
                                               context_mask=context_mask)

        x = x + attn_out
        context = context + context_attn_out

        x = self.ff(x) + x
        context = self.context_ff(context) + context

        return x, context
