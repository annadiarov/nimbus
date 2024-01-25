# Data sources

Herein we describe the data sources used in this project.

## pHLA binding data

### NetMHCpan 4.1

Data from Reynisson et al. (2020) is publicly available at the [supplementary 
information](https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/) 
associated with the publication. 

We provided the data compressed, you can decompress it following the indication
of the supplementary information:
```bash
cat NetMHCpan_train.tar.gz | uncompress | tar xvf -
cat CD8_benchmark.tar.gz | uncompress | tar xvf -
```

Reynisson, B.; Alvarez, B.; Paul, S.; Peters, B.; Nielsen, M. NetMHCpan-4.1
and NetMHCIIpan-4.0: Improved Predictions of MHC Antigen Presentation by 
Concurrent Motif Deconvolution and Integration of MS MHC Eluted Ligand Data.
_Nucleic Acids Res._ **2020**, 48, W449–W454.
[![DOI:10.1093/nar/gkaa379](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1093/nar/gkaa379)


<details>

<summary><strong>Data description</strong></summary>

We provide the raw data from the supplementary information. It contains:

- `NetMHCpan_train.tar.gz`: Training data for NetMHCpan 4.1
- `CD8_benchmark.tar.gz`: Test data for NetMHCpan 4.1

> #### Training data
> The training data is composed of 12 files:
> - `MHC_pseudo.dat`: MHC allele information. It has the following format:
>   ```text
>     H2-Lq YESYYRIIAGQWFVNTLYIRYEYYTWAAYAYEWY
>     HLA-A0101 YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY
>     HLA-A0102 YSAMYQENMAHTDANTLYIIYRDYTWVARVYRGY
>   ```
>   Where the **first column is the allele name** and the **second column is the
>   MHC Pseudo sequence** (includes residues 34 MHC-1 amino acids selected as 
>   the polimorfic residues in contact with the peptide. Source
>   [Nielsen et al. (2007)](https://doi.org/10.1371/journal.pone.0000796)).
> - `allelelist`: List of alleles used in the training data. It has the following
>  format:
>  ```text
>   # For multi-allelic Eluted Ligand data
>   EBL BoLA-3:01101,BoLA-2:04801,gb1.7,BoLA-3:05001,amani.1
>   Fibroblast	HLA-A03:01,HLA-A23:01,HLA-B08:01,HLA-B15:18,HLA-C07:02,HLA-C07:04
>   GD149	HLA-A01:01,HLA-A24:02,HLA-B38:01,HLA-B44:03,HLA-C06:02,HLA-C12:03
>   JY	HLA-A02:01,HLA-B07:02,HLA-C07:02
>   ...
>   # Single Allele from Eluted Ligand mono-allelic data and binding affinity data
>   H2-Ld H2-Ld
>   H2-Lq H2-Lq
>   HLA-A01:01 HLA-A01:01
>   HLA-A01:03 HLA-A01:03
>   ```
>  Where the **first column is the cell line name** (for Multi-allelic data) **or
>  the MHC allele** (for single allele data) while the **second column is the
>  list of alleles**. 
> 
>  ⚠️ The names in the first column are the ones that will appear in `coo*_*` 
>  files (below), which contain peptide-MHC pairs with their affinity. When the 
>  pair comes from a Multi-allelic experiment, we will find the cell line name
>  instead of the MHC allele name.
> - 5 files (`c00*_ba`) containing binding affinity data for different length peptides
>   - 5 files (`c00*_el`) containing eluted ligand data for different length peptides
> 
>     These 10 files have the following format:
>     ```text
>     # Binding affinity sample data
>     AAFTNHNYI 0.439534 H-2-Kb
>     AAFTNHNYI 0.806158 H-2-Db
>     AAGIGILTV 0.550126 HLA-A02:01
>     AAGIGILTVI 0.203084 HLA-A02:01
>     AAHARFVAA 0.0545091 HLA-A11:01
>     AAHARFVAA 0.171722 HLA-A24:02
>     # Eluted ligand sample data
>     TEAARELGY 1 HLA-B44:03
>     ATDYPLIAR 1 pat-FL
>     RQPDSGISSI 1 pat-NS2
>     SVDIDSEL 1 Line.27
>     DGDEDLPGPPVRYY 1 HLA-A01:01
>     ```
>     The **first column is the peptide sequence**, the **second column is the binding
>     affinity (μM)** and the **third column is the HLA allele**. The **eluted ligand data**
>     is the **same, but the second column is the number of times the peptide was
>     observed in the eluted ligand data** and in the **third column** we will find
>     the **cell line _when_ the data comes from multi-allelic assays**.
>
>     True binders are those with a binding affinity below 500 nM (0.5 μM, threshold used
>     in the publication) and eluted ligands above 1.
>   
>    ⚠️ Binding affinity data with affinity = 0.0 μM are artificial negative peptides.
>    Similarly happens with eluted ligand data with 0 observed peptides. See 
>    section below for more information about negative peptides.
>   
>     ##### Data insights from the publication
>     All peptides employed in the training were filtered to only include 8 to 14 amino acid long peptides. All
>     MHCs present in the Binding Affinity (BA) subset were enriched with 100 random negative sequences (target value of 
>     0.01). On the other hand, positive peptides for each MHC present in the EL subset were enriched, 
>     length-wise, with 5 times the amount of peptides of the most abundant peptide length.
>     More in detail, the amount of random negatives was imposed to be the same for 
>     each length 8–13, and corresponded for each length to five times the amount 
>     of positives for the most abundant peptide length. This uniform length distribution 
>     of the random negatives was adopted as a background against which machine 
>     learning can be employed to learn the amino acid and length preference of the natural binders.
>     (Soruce: [Alvarez et al. (2018)](https://doi.org/10.1002/pmic.201700252))
> 
>    ###### Generation of negative peptides
>    As described in [Jurtz et al. (2017)](https://doi.org/10.4049/jimmunol.1700893),
>    negative peptides were generated by randomly sampling 8-14 aminoacids peptides
>    from the source antigen protein (using UniProt sequences)
>   

> #### Test data
> Under construction
>
>     ##### Data insights from the publication
> 
</details>

## pHLA stability data

### NetMHCstabpan
Data from Rasmussen et al. (2016) was provided by Morten Nielsen, so we won't 
include it in this repository.

**WARNING**: This data is only for academic use. Please contact the authors of the
original publication for commercial use.

Rasmussen, M.; Fenoy, E.; Harndahl, M.; Kristensen, A. B.; Nielsen, I. K.;
Nielsen, M.; Buus, S. Pan-Specific Prediction of Peptide–MHC Class I Complex
Stability, a Correlate of T Cell Immunogenicity. *J. Immunol.* **2016**, 197, 1517–1524.
[![DOI:10.4049/jimmunol.1600582](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.4049/jimmunol.1600582)

## TCR-pMHC recognition data



## Neoantigen immunogenicity data

