import torch
import numpy as np
from nimbus.data_processing import SeqTokenizer


class pHLADataset(torch.utils.data.Dataset):
    def __init__(self,
                 peptide_seq_arr: np.ndarray,
                 hla_names_arr: np.ndarray,
                 hla_fp_dict: dict,
                 labels: np.ndarray,
                 max_peptide_len: int = 15):
        # Validate inputs
        assert len(peptide_seq_arr) == len(hla_names_arr) == len(labels), \
            'Peptide, HLA and labels must have the same length'
        assert type(peptide_seq_arr) == np.ndarray, \
            'peptide_seq_arr must be a numpy array'
        assert type(hla_names_arr) == np.ndarray, \
            'hla_names_arr must be a numpy array'
        assert type(labels) == np.ndarray, 'labels must be a numpy array'
        hla_keys = set(list(hla_fp_dict))
        unique_hla_list = set(hla_names_arr)
        if not unique_hla_list.issubset(hla_keys):
            err_msg = (f"hla_fp_dict does not have all the hla_names."
                       f" Missing HLAs: {unique_hla_list - hla_keys}")
            raise KeyError(err_msg)
        pep_tokenizer = SeqTokenizer()
        peptide_pad = \
            [pep + pep_tokenizer.padding_token[0] * (max_peptide_len - len(pep))
             for pep in peptide_seq_arr]
        peptide_idx_list = [torch.Tensor(pep_tokenizer.encode(pep)) for
                            pep in peptide_pad]
        self.peptides = peptide_idx_list
        hlas_fp_list = [torch.Tensor(hla_fp_dict[hla]) for hla in hla_names_arr]
        self.hlas = hlas_fp_list  # In the same order as peptide
        self.labels = torch.Tensor(labels)  # 1D array of 0s and 1s

    def __len__(self):
        return len(self.hlas)

    def __getitem__(self, idx):
        hla_fp = self.hlas[idx]
        peptide_idx = self.peptides[idx]
        label = self.labels[idx]
        return peptide_idx, hla_fp, label
