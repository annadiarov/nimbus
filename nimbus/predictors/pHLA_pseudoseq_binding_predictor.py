import io
import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import roc_curve, auc
from einops.layers.torch import Rearrange, Reduce
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from nimbus.utils import CosineWarmupScheduler
from nimbus.ml_blocks import SelfAttentionBlock, JointCrossAttentionBlock, FILIPBlock
from nimbus.globals import DEVICE



class pHLAPseudoseqBindingPredictor(L.LightningModule):
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
        HLA pseudo-sequence.
        :param lr: float
            Learning rate
        :param warmup: int
            Number of lr-warmup iterations
        :param n_iterations_cosine_cycle: int
            Number of optimization iterations in a cosine cycle (not epoch!)
        :param dropout: float
            Dropout rate for self-attention, cross-attention, and FILIP layers
        :param hla_n_fp: int
            HLA pseudo-sequence length
        :param hla_fp_dim: int
            HLA pseudo-sequence embedding dimension
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
        # Used in testing mode
        self._auroc = BinaryAUROC()
        self._test_outputs = []  # Initialize a list to store test outputs

    def _create_model(self):
        # Dynamic (learnable) positional encodings
        self.h_pe = nn.Parameter(
            torch.empty(self.hparams.hla_n_fp, self.hparams.hla_fp_dim)
        )
        self.p_pe = nn.Parameter(
            torch.empty(self.hparams.pep_seq_len, self.hparams.pep_embedding_dim)
        )

        # Peptide sequence embedding: N,15,1 -> N,15,32
        self.p_embedding = nn.Embedding(21,  # 20 amino acids + 1 padding
                                        self.hparams.pep_embedding_dim)
        self.h_embedding = nn.Embedding(21,  # 20 amino acids + 1 padding
                                        self.hparams.hla_fp_dim)
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
            *[JointCrossAttentionBlock(
                dim=self.hparams.hla_fp_dim,
                context_dim=self.hparams.pep_embedding_dim,
                dropout=self.hparams.dropout
             ) for _ in range(self.hparams.n_joint_cross_attns)
            ]
        )

        self.filip = FILIPBlock(
            dim=self.hparams.hla_fp_dim,
            context_dim=self.hparams.pep_embedding_dim,
            heads=self.hparams.filip_num_heads,
            dim_head=self.hparams.filip_dim_head,
            dropout=self.hparams.dropout
        )

        self.linear_to_logits = nn.Linear(
            self.hparams.filip_num_heads,
            self.hparams.filip_num_heads
        )

        self.to_pred = nn.Sequential(
            Reduce('... n d -> ... d', 'mean'),
            nn.LayerNorm(self.hparams.filip_num_heads),
            nn.Linear(self.hparams.filip_num_heads, 1),
            Rearrange('... 1 -> ...')
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.normal_(self.p_embedding.weight, std=0.02)
        nn.init.normal_(self.h_embedding.weight, std=0.02)
        nn.init.normal_(self.p_pe, std=0.01)
        nn.init.normal_(self.h_pe, std=0.01)

    def forward(self, p, h, save_attn=False):
        # if save_attn:
        #     pep_self_attn, hla_self_attn = [], []
        #     pep_cross_attn, hla_cross_attn = [], []
        if save_attn:
            pep_self_attn_matrices, hla_self_attn_matrices = [], []
            pep_cross_attn_matrices, hla_cross_attn_matrices = [], []

        h = h.squeeze().int()
        h = self.h_embedding(h)
        h = h + self.h_pe
        h = self.h_ln(h)

        if save_attn:
            for h_self_attn in self.h_self_attns:
                h, h_self_attn_mat = h_self_attn(h, return_attn=save_attn)
                hla_self_attn_matrices.append(
                    h_self_attn_mat.detach().cpu().numpy())
        else:
            for h_self_attns in self.h_self_attns:
                h = h_self_attns(h)

        p = p.squeeze().int()
        p = self.p_embedding(p)
        p = p + self.p_pe
        p = self.p_ln(p)
        if save_attn:
            for p_self_attn in self.p_self_attns:
                p, p_self_attn_mat = p_self_attn(p, return_attn=save_attn)
                pep_self_attn_matrices.append(p_self_attn_mat.detach().cpu().numpy())
        else:
            for p_self_attns in self.p_self_attns:
                p = p_self_attns(p)

        h_mask = torch.ones((1, self.hparams.hla_n_fp)).bool().to(DEVICE)
        p_mask = torch.ones((1, self.hparams.pep_seq_len)).bool().to(DEVICE)

        if save_attn:
            for cross_attn in self.joint_cross_attns:
                h, p, h_attn, p_attn = cross_attn(
                    h,
                    context=p,
                    mask=h_mask,
                    context_mask=p_mask,
                    return_attn=save_attn
                )
                hla_cross_attn_matrices.append(h_attn.detach().cpu().numpy())
                pep_cross_attn_matrices.append(p_attn.detach().cpu().numpy())

        else:
            for cross_attn in self.joint_cross_attns:
                h, p = cross_attn(
                    h,
                    context=p,
                    mask=h_mask,
                    context_mask=p_mask
                )

        if save_attn:
            x, filip_interactions = (
                self.filip(h, p, context_mask=p_mask,
                           save_raw_interactions=save_attn))
            filip_interactions = filip_interactions.detach().cpu().numpy()
        else:
            x = self.filip(h, p, context_mask=p_mask, save_raw_interactions=save_attn)

        if save_attn:
            attns_dict = {
                'pep_self_attn': pep_self_attn_matrices,
                'hla_self_attn': hla_self_attn_matrices,
                'pep_cross_attn': pep_cross_attn_matrices,
                'hla_cross_attn': hla_cross_attn_matrices,
                'filip_interactions': filip_interactions,
            }
            return x, attns_dict
        return x

    def _calculate_loss(self, batch, mode="train"):
        peptide_data, hla_data, labels = batch
        batch_pos_idx = torch.where(labels == 1.)
        batch_neg_idx = torch.where(labels == 0.)
        # Perform prediction and calculate loss and accuracy
        logits = self.forward(peptide_data, hla_data)
        logits = self.linear_to_logits(logits)
        logits = self.to_pred(logits)

        target = labels.float()

        loss = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        pos_prob = probs[batch_pos_idx].mean()
        neg_prob = probs[batch_neg_idx].mean()
        acc = ((logits > 0).long() == labels).float().sum() / len(labels)
        # calculate precision, recall, f1
        pos_precision = torch.sum((logits > 0).long() * labels) / torch.sum(logits > 0).float()
        pos_recall = torch.sum((logits > 0).long() * labels) / torch.sum(labels).float()
        pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
        neg_precision = torch.sum((logits <= 0).long() * (1 - labels)) / torch.sum(logits <= 0).float()
        neg_recall = torch.sum((logits <= 0).long() * (1 - labels)) / torch.sum(1 - labels).float()
        neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        # Logging as dict
        self.log_dict({f"{mode}_loss": loss,
                       f"{mode}_acc": acc,
                       f"{mode}_pos_precision": pos_precision,
                       f"{mode}_pos_recall": pos_recall,
                       f"{mode}_pos_f1": pos_f1,
                       f"{mode}_neg_precision": neg_precision,
                       f"{mode}_neg_recall": neg_recall,
                       f"{mode}_neg_f1": neg_f1,
                       f"{mode}_pos_prob": pos_prob,
                       f"{mode}_neg_prob": neg_prob},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss, acc, probs

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
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _, _, probs = self._calculate_loss(batch, mode="test")
        _, _, labels = batch
        self._test_outputs.append({'probs': probs, 'labels': labels})
        return {'probs': probs, 'labels': labels}

    def on_test_epoch_end(self):
        # Concatenate all probabilities and labels from outputs
        all_probs = torch.cat([x['probs'] for x in self._test_outputs])
        all_labels = torch.cat([x['labels'] for x in self._test_outputs]).int()

        # Compute ROC-AUC score
        roc_auc_score = self._auroc(all_probs, all_labels)
        self.log('test_roc_auc', roc_auc_score)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(all_labels.cpu(), all_probs.cpu())
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Save figure to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert buffer to PIL Image and then to numpy array
        img = Image.open(buf)
        img_array = np.array(img)

        # Log ROC curve figure to TensorBoard
        self.logger.experiment.add_image('ROC Curve', img_array, global_step=self.global_step, dataformats='HWC')

        # Reset the metric for the next epoch
        self._auroc.reset()

        # Clear the list for the next epoch
        self._test_outputs.clear()
