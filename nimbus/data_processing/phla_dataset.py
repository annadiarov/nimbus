import torch
import numpy as np
from nimbus.data_processing import SeqTokenizer


class pHLADataset(torch.utils.data.Dataset):
    def __init__(self,
                 peptide_seq_arr: np.ndarray,
                 hla_names_arr: np.ndarray,
                 hla_fp_dict: dict,
                 labels: np.ndarray,
                 max_peptide_len: int = 15,
                 has_augmented_hla: bool = False,
                 use_all_augmented_data: bool = False):
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
        if has_augmented_hla and use_all_augmented_data:
            # if no augmented data, hla_fp_dict has hla_names as key and fingerprint as torch 2d array in dict values
            #  If the data is augmented, instead of 2d array, we have a 3d array where the first dim is the number of different fingerprints for that hla_names
            # If data augmented is provided, then expand the number of peptides (and the label for each pair) by the number of different fingerprints for that allele
            self.peptides = []
            self.hlas = []
            self.labels = []
            for i, hla in enumerate(hla_names_arr):
                hla_fp = hla_fp_dict[hla]
                num_fp = hla_fp.shape[0]  # Number of different fingerprints for this allele
                for j in range(num_fp):
                    self.peptides.append(peptide_idx_list[i])
                    self.hlas.append(torch.Tensor(hla_fp[j]))
                    self.labels.append(labels[i])
            self.labels = torch.Tensor(self.labels)
        elif has_augmented_hla and not use_all_augmented_data:
            # For each pair of peptide and hla, randomly select one hla fingerprint
            self.peptides = peptide_idx_list
            self.hlas = []
            self.labels = []
            for i, hla in enumerate(hla_names_arr):
                hla_fp = hla_fp_dict[hla]
                num_fp = hla_fp.shape[0]
                idx = np.random.randint(num_fp)
                self.hlas.append(torch.Tensor(hla_fp[idx]))
            self.labels = torch.Tensor(labels)
        else:
            self.peptides = peptide_idx_list
            hlas_fp_list = [torch.Tensor(hla_fp_dict[hla]) for hla in hla_names_arr]
            self.hlas = hlas_fp_list  # In the same order as peptide
            self.labels = torch.Tensor(labels)  # 1D array of 0s and 1s

        assert len(self.peptides) == len(self.hlas) == len(self.labels), \
            'Peptide, HLA and labels must have the same length'

    def __len__(self):
        return len(self.hlas)

    def __getitem__(self, idx):
        hla_fp = self.hlas[idx]
        peptide_idx = self.peptides[idx]
        label = self.labels[idx]
        return peptide_idx, hla_fp, label
