"""
Feed Forward Block extracted from the following repository:
    https://github.com/lucidrains/tf-bind-transformer
"""
import torch.nn as nn


def FeedForwardBlock(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )
