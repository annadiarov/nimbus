"""
Code for Cross-Attention Block extracted from the following repository:
    https://github.com/lucidrains/tf-bind-transformer
"""
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from nimbus.utils import exists

# FILIP adapter model
from logavgexp_pytorch import logavgexp


# helper functions
def l2norm(t):
    return F.normalize(t, dim=-1)


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


class FILIPBlock(nn.Module):
    def __init__(
            self,
            dim,
            context_dim,
            heads,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        inner_latent_dim = heads * dim_head

        self.to_latent_w = nn.Parameter(torch.randn(dim, inner_latent_dim))
        self.to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.pre_attn_dropout = dropout

        self.null_context = nn.Parameter(torch.randn(heads, dim_head))
        self.context_to_latent_w = nn.Parameter(
            torch.randn(context_dim, inner_latent_dim))
        self.context_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

    def forward(
            self,
            x,
            context,
            context_mask=None
    ):
        b, heads, device = x.shape[0], self.heads, x.device

        # n: sequence length or n_hla_fp, d: sequence emb dim
        x = einsum('b n d, d e -> b n e', x, self.to_latent_w)
        x = x + self.to_latent_b

        # h: n_head, d: dim_head
        x = rearrange(x, 'b n (h d) -> b h n d', h=heads)

        context = einsum('b n d, d e -> b n e', context, self.context_to_latent_w)
        context = context + self.context_to_latent_b

        context = rearrange(context, 'b n (h d) -> b h n d', h=heads)

        context, x = map(l2norm, (context, x))

        # fine grained interaction between dna and protein sequences
        # FILIP https://arxiv.org/abs/2111.07783

        if x.shape[0] == 1:
            # in the case one passes in 1 genomic sequence track
            # but multiple factors + contexts, as in enformer training
            x = rearrange(x, '1 ... -> ...')
            einsum_eq = 'h i d, b h j d -> b h i j'
        else:
            einsum_eq = 'b h i d, b h j d -> b h i j'

        # create context mask if not exist

        if not exists(context_mask):
            context_mask = torch.ones((b, context.shape[-1]),
                                      device=device).bool()

        # dropout mask by dropout prob

        if self.training:
            keep_mask = prob_mask_like(context_mask, 1 - self.pre_attn_dropout)
            context_mask = context_mask & keep_mask

        # add null context and modify mask (provides a learnable representation
        #  for the case where no relevant context is available.

        context_mask = F.pad(context_mask, (1, 0), value=True)
        context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        null_context = repeat(self.null_context, 'h d -> b h 1 d', b=b)
        context = torch.cat((null_context, context), dim=-2)

        # differentiable max, as in FILIP paper

        interactions = einsum(einsum_eq, x, context)  # B, H, X, C+1
        interactions = logavgexp(interactions, mask=context_mask, dim=-1,
                                 temp=0.05)  # B, H, X where X is input dim
        interactions = rearrange(interactions, 'b h i -> b i h')
        return interactions
