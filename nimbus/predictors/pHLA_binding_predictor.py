import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from nimbus.utils import CosineWarmupScheduler
from nimbus.ml_blocks import SelfAttentionBlock, JointCrossAttentionBlock, FILIPBlock
from nimbus.globals import DEVICE


class pHLABindingPredictor(L.LightningModule):
    def __init__(
            self,
            lr: float = 1e-3,
            warmup: int = 20,
            n_iterations_cosine_cycle: int = 100,
            dropout: float = 0.1,
            hla_n_fp: int = 400,
            hla_fp_dim: int = 80,
            pep_seq_len: int = 15,
            pep_embedding_dim: int = 32,
            n_self_attns: int = 2,
            n_joint_cross_attns: int = 4,
            filip_num_heads: int = 64,
            filip_dim_head: int = 16,
            **kwargs):
        """
        Predictor for pHLA binding affinity given peptide sequence and
        HLA allele surface fingerprint.
        :param lr: float
            Learning rate
        :param warmup: int
            Number of lr-warmup iterations
        :param n_iterations_cosine_cycle: int
            Number of optimization iterations in a cosine cycle (not epoch!)
        :param dropout: float
            Dropout rate for self-attention, cross-attention, and FILIP layers
        :param hla_n_fp: int
            Number of Fingerprints for each HLA
        :param hla_fp_dim: int
            Dimension of each fingerprint
        :param pep_seq_len: int
            Length of peptide sequence
        :param pep_embedding_dim: int
            Dimension of peptide embedding
        :param n_self_attns: int
            Number of self-attention layers for peptide and HLA
        :param n_joint_cross_attns: int
            Number of joint cross-attention layers
        :param filip_num_heads: int
            Number of heads for the FILIP layer
        :param filip_dim_head: int
            Dimension of each head for the FILIP layer
        :param kwargs:
            Added to accept any additional arguments in config_dicts, but they
            are not used
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _create_model(self):
        # Dynamic (learnable) positional encodings
        self.h_pe = nn.Parameter(torch.empty(self.hparams.hla_n_fp,
                                             self.hparams.hla_fp_dim))
        self.p_pe = nn.Parameter(torch.empty(self.hparams.pep_seq_len,
                                             self.hparams.pep_embedding_dim))

        # Peptide sequence embedding: N,15,1 -> N,15,32
        self.p_embedding = nn.Embedding(21,  # 20 amino acids + 1 padding
                                        self.hparams.pep_embedding_dim)

        self.h_ln = nn.LayerNorm(self.hparams.hla_fp_dim)
        self.p_ln = nn.LayerNorm(self.hparams.pep_embedding_dim)

        self.h_self_attns = nn.Sequential(
            *[SelfAttentionBlock(dim=self.hparams.hla_fp_dim,
                                 dropout=self.hparams.dropout)
              for _ in range(self.hparams.n_self_attns)])

        self.p_self_attns = nn.Sequential(
            *[SelfAttentionBlock(dim=self.hparams.pep_embedding_dim,
                                 dropout=self.hparams.dropout)
              for _ in range(self.hparams.n_self_attns)])

        self.joint_cross_attns = nn.Sequential(
            *[JointCrossAttentionBlock(dim=self.hparams.hla_fp_dim,
                                       context_dim=self.hparams.pep_embedding_dim,
                                       dropout=self.hparams.dropout)
              for _ in range(self.hparams.n_joint_cross_attns)])

        self.filip = FILIPBlock(
            dim=self.hparams.hla_fp_dim,
            context_dim=self.hparams.pep_embedding_dim,
            heads=self.hparams.filip_num_heads,
            dim_head=self.hparams.filip_dim_head,
            dropout=self.hparams.dropout
        )

        self.linear_to_logits = nn.Linear(self.hparams.filip_num_heads,
                                          self.hparams.filip_num_heads)

        self.to_pred = nn.Sequential(
            Reduce('... n d -> ... d', 'mean'),
            nn.LayerNorm(self.hparams.filip_num_heads),
            nn.Linear(self.hparams.filip_num_heads, 1),
            Rearrange('... 1 -> ...')
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.normal_(self.p_embedding.weight, std=0.02)
        nn.init.normal_(self.p_pe, std=0.01)
        nn.init.normal_(self.h_pe, std=0.01)

    def forward(self, x):
        h, p = x[:, :, :-15], x[:, :, -15:]

        h = h.reshape(-1, 400, 80)
        h = h + self.h_pe
        h = self.h_ln(h)
        for h_self_attns in self.h_self_attns: h = h_self_attns(h)
        p = p.squeeze().int()
        p = self.p_embedding(p)
        p = p + self.p_pe
        p = self.p_ln(p)
        for p_self_attns in self.p_self_attns: p = p_self_attns(p)

        h_mask = torch.ones((1, self.hparams.hla_n_fp)).bool().to(DEVICE)
        p_mask = torch.ones((1, self.hparams.pep_seq_len)).bool().to(DEVICE)

        for cross_attn in self.joint_cross_attns:
            h, p = cross_attn(h, context=p, context_mask=p_mask)

        x = self.filip(h, p, context_mask=p_mask)

        return x

    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        batch_pos_idx = torch.where(labels == 1.)
        batch_neg_idx = torch.where(labels == 0.)
        # Perform prediction and calculate loss and accuracy
        logits = self.forward(inp_data)
        logits = self.linear_to_logits(logits)
        logits = self.to_pred(logits)

        target = labels.float()

        loss = F.binary_cross_entropy_with_logits(logits, target)
        pos_prob = torch.sigmoid(logits[batch_pos_idx]).mean()
        neg_prob = torch.sigmoid(logits[batch_neg_idx]).mean()
        acc = ((logits > 0).long() == labels).float().sum() / len(labels)
        # Logging
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log(f"{mode}_acc", acc, on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log(f"{mode}_pos_prob", pos_prob, on_step=False, on_epoch=True,
                 prog_bar=True)
        self.log(f"{mode}_neg_prob", neg_prob, on_step=False, on_epoch=True,
                 prog_bar=True)

        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # We don't return the lr scheduler because we need to apply it per
        # iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams.warmup,
            n_iterations_cycle=self.hparams.n_iterations_cosine_cycle
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")
