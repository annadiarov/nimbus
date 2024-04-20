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

> #### Train data
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
>  ⚠️ The names in the first column are the ones that will appear in `c00*_*` 
>  files (below), which contain peptide-MHC pairs with their affinity. When the 
>  pair comes from a Multi-allelic experiment, we will find the cell line name
>  instead of the MHC allele name.
> - 5 files (`c00*_ba`) containing binding affinity data for different length peptides
> - 5 files (`c00*_el`) containing eluted ligand data for different length peptides
> 
>  These 10 files have the following format:
>  ```text
>  # Binding affinity sample data
>  AAFTNHNYI 0.439534 H-2-Kb
>  AAFTNHNYI 0.806158 H-2-Db
>  AAGIGILTV 0.550126 HLA-A02:01
>  AAGIGILTVI 0.203084 HLA-A02:01
>  AAHARFVAA 0.0545091 HLA-A11:01
>  AAHARFVAA 0.171722 HLA-A24:02
>  # Eluted ligand sample data
>  TEAARELGY 1 HLA-B44:03
>  ATDYPLIAR 1 pat-FL
>  RQPDSGISSI 1 pat-NS2
>  SVDIDSEL 1 Line.27
>  DGDEDLPGPPVRYY 1 HLA-A01:01
>  ```
>  The **first column is the peptide sequence**, the **second column is the binding
>  affinity _scaled_ between 0 and 1** and the **third column is the HLA allele**. The **eluted ligand data**
>  is the **same, but the second column is the number of times the peptide was
>  observed in the eluted ligand data** and in the **third column** we will find
>  the **cell line _when_ the data comes from multi-allelic assays**.
>
>  True binders were considered those with a binding affinity below 500 nM (0.5 μM, threshold used
>  in the publication) and eluted ligands above 1.
>  ⚠️ For binding affinity data, the threshold for the scaled affinities
>  seems to be scaled_binding_affinity >= 0.426 for binders. Notice this value was a
>  manually obtained threshold to get the same number of binders that are in the
>  paper supplementary data.
>   
> ⚠️ Binding affinity data with affinity = 0.0 μM are artificial negative peptides.
> Similarly happens with eluted ligand data with 0 observed peptides. See 
> section below for more information about negative peptides.
>   
>  ##### Data insights from the publication
>  All peptides employed in the training were filtered to only include 8 to 14 amino acid long peptides. All
>  MHCs present in the Binding Affinity (BA) subset were enriched with 100 random negative sequences (target value of 
>  0.01). On the other hand, positive peptides for each MHC present in the EL subset were enriched, 
>  length-wise, with 5 times the amount of peptides of the most abundant peptide length.
>  More in detail, the amount of random negatives was imposed to be the same for 
>  each length 8–13, and corresponded for each length to five times the amount 
>  of positives for the most abundant peptide length. This uniform length distribution 
>  of the random negatives was adopted as a background against which machine 
>  learning can be employed to learn the amino acid and length preference of the natural binders.
>  (Soruce: [Alvarez et al. (2018)](https://doi.org/10.1002/pmic.201700252))
> 
> ###### Generation of negative peptides
> As described in [Jurtz et al. (2017)](https://doi.org/10.4049/jimmunol.1700893),
> negative peptides were generated by randomly sampling 8-14 aminoacids peptides
> from the source antigen protein (using UniProt sequences)
>   

> #### Test data
> Contains 1660 epitopes from 52 different MHC-1 alleles ([Jurtz et al. (2017)](https://doi.org/10.4049/jimmunol.1700893))
> This dataset was limited by the HLA molecules covered by the methods included 
> in their benchmark (NetMHCpan-4.1, NetMHCpan-4.0, MixMHCpred, MHCFlurry,
> MHCFlurry_EL).
> 
> In the compressed file there are 1660 files, one per epitope. Each file has the
> following format:
> ```text
> # Epitope sequence
> THSFEFAQFDNFLV 0 HLA-A02:01
> HSFEFAQFDNFLVE 0 HLA-A02:01
> SFEFAQFDNFLVEA 0 HLA-A02:01
> FEFAQFDNFLVEAT 0 HLA-A02:01
> EFAQFDNFLVEATR 0 HLA-A02:01
> YVVTWIVGA 1 HLA-A02:01
> ```
> Where the **first column is the peptide sequence**, the **second column is the
> binding label** (0 non-binder, 1 binder. In each document we will have a single
> binder and the rest are non-binders) and the **third column is the HLA allele**.
> 
> There is also a file called `CD8_mapped`, which contains the following information:
> ```text
> 1 AAAGAAVTV HLA-A02:01 MPVDSSSTHRHRCVAAPLVRLAAAGAAVTVAVGTAAAWAHAGAPQHRCIHDAMQARVLQSVAAQRMAPSAVSAVGLPYVSVVPVENASTLDYSLSDSTSPGVVRAANWGALRVAVSAEDLTDPAYHCARVGQQVNNHAGDIVTCTAEDILTDEKRDTLVKHLVPQALQLHRERLKVRQVQGKWKVTGMADVICGDFKVPPEHITEGVTNTDFVLYVASVPSEESVLAWATTCQVFPDGHPAVGVINIPAANIASRYDQLVTRVVTHEMAHAVGFSGTFFGAVGIVQEVPHLRRKDFNVSVITSSTVVAKAREQYGCNSLEYLEIEDQGGAGSAGSHIKMRNAKDELMAPAASAGYYTALTMAVFQDLGFYQADFSKAEEMPWGRNVGCAFLSEKCMAKNVTKWPAMFCNESAATIRCPTDRLRVGTCGITAYNTSLATYWQYFTNASLGGYSPFLDYCPFVVGYRNGSCNQDASTTPDLLAAFNVFSEAARCIDGAFTPKNRTAADGYYTALCANVKCDTATRTYSVQVRGTNGYANCTPGLRVKLSSVSDAFEKGGYVTCPPYVEVCQGNVKAAKDFAGDTDSSSSADDAADKEAMQRWSDRMAALATATTLLLGMVLSLMALLVVRLLLTSSPWCCCRLGGLPT  AAAGAAVTV_HLA-A02:01
> 2 AAAGFLFCV HLA-A02:01 MKGGCVSQWKAAAGFLFCVMVFASAERPVFTNHFLVELHKGGEDKARQVAAEHGFGVRKLPFAEGLYHFYHNGLAKAKRRRSLHHKQQLERDPRVKMALQQEGFDRKKRGYRDINEIDINMNDPLFTKQWYLINTGQADGTPGLDLNVAEAWELGYTGKGVTIGIMDDGIDYLHPDLASNYNAEASYDFSSNDPYPYPRYTDDWFNSHGTRCAGEVSAAANNNICGVGVAYNSKVAGIRMLDQPFMTDIIEASSISHMPQLIDIYSASWGPTDNGKTVDGPRELTLQAMADGVNKGRGGKGSIYVWASGDGGSYDDCNCDGYASSMWTISINSAINDGRTALYDESCSSTLASTFSNGRKRNPEAGVATTDLYGNCTLRHSGTSAAAPEAAGVFALALEANLGLTWRDMQHLTVLTSKRNQLHDEVHQWRRNGVGLEFNHLFGYGVLDAGAMVKMAKDWKTVPERFHCVGGSVQDPEKIPSTGKLVLTLTTDACEGKENFVRYLEHVQAVITVNATRRGDLNINMTSPMGTKSILLSRRPRDDDSKVGFDKWPFMTTHTWGEDARGTWTLELGFVGSAPQKGVLKEWTLMLHGTQSAPYIDQVVRDYQSKLAMSKKEELEEELDEAVERSLKSILNKN   AAAGFLFCV_HLA-A02:01
> ...
> ```
> Where the **first column is the peptide sequence**, the **second column is the
> HLA allele**, the **third column is MHC sequence** and the **fourth column is the
> name of the file** where you can find that peptide with its negative peptides.
> 
> ##### Data insights from the publication
> This data set consists of the epitope data set from Jurtz et al. combined with multimer validated 
> epitopes obtained from the IEDB (downloaded 11-04-2020). The data set was filtered to only contain 
> epitopes of length 8-14, mapped to fully typed HLA molecules covered by all methods included in the 
> benchmark, and annotated source protein sequence.
> 
> Additional SA EL datasets were downloaded. Each dataset SA was 
> enriched, length-wise, with negative decoy peptides of 5 times the amount of ligands of the most 
> abundant peptide length.
> 
> Finally, to ensure the independent test set’s orthogonality, 
> positive peptides overlapping with the training data were removed from all test sets. The resulting 
> benchmark datasets consisted of 1,660 epitopes restricted to 52 distinct MHC-I molecules, and 36 SA 
> EL datasets covering a total 45,416 MS MHC eluted ligands.
</details>

### MixMHCpred 2.2

Data from Gfeller et al. (2023) is publicly available at the [supplementary
information](https://www.nature.com/articles/s41586-021-03819-2#Sec19) associated
with the publication.

According to the license, this data is only for academic use. Please contact the
authors of the original publication for commercial use.

Gfeller, D.; Schmidt, J.; Croce, G.; Guillaume, P.; Bobisse, S.; Genolet, R.; 
Queiroz, L.; Cesbron, J.; Racle, J.; Harari, A. Improved Predictions of Antigen 
Presentation and TCR Recognition with MixMHCpred2.2 and PRIME2.0 Reveal Potent 
SARS-CoV-2 CD8+ T-Cell Epitopes. _Cell Syst._ **2023**, 14, 72-83.e5. 
[![DOI:10.1016/j.cels.2022.12.002](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1016/j.cels.2022.12.002)

<details>

<summary><strong>Data description</strong></summary>

We provide the raw data from the supplementary information. It contains:

> #### Train data
> The training data is contained in the file `train_2023_MixMHCpred2.2.txt`. It
> corresponds to the Table S2 in the supplementary information, and it has the 
> list of HLA-I ligands used to train MixMHCpred2.2
> 
> The file has the following format:
> ```text
> Peptide	Allele
> AAAHTHRY	A0101
> ADMGHLKY	A0101
> AIDNPLHY	A0101
> ...
> ```
> Where the **first column is the peptide sequence** and the **second column is the
> HLA allele**.
> 
> ##### Data insights from the publication
> They collected data from 24 different sources (see Table S1 in the publication
> supplementary information). 
> All peptides-HLA alleles were processed with the motif deconvolution tool MixMHCp.
> MixMHCp is an unsupervised algorithm that assigns peptide from MS peptidomic 
> data to its cognate HLA allale. Samples with low confidence were removed from the
> training set by the authors.
> 
> This resulted in a training set covering 119 HLA-I alleles, supported by a 
> total of 384,070 peptides 
> 
> ⚠️ Notice that this training set does not include negative peptides, since
> MixMHCpred2.2 is an heuristic method that does not require negative peptides.

> #### Test data
> The test data is contained in the folder `val_2023_MixMHCpred2.2`. It
> corresponds to the Table S3 in the supplementary information.
> 
> The data was transformed from the original xlsx to a CSV as follows:
> ```python
> import os
> import pandas as pd # requires optinal dependeciy openpyxl
>
> def excel_to_csv(input_excel_file, output_csv_folder):
>    # Read the Excel file
>    xls = pd.ExcelFile(input_excel_file)
>
>    # Iterate through each sheet in the Excel file
>    for sheet_name in xls.sheet_names:
>        # Read the sheet into a DataFrame
>        df = xls.parse(sheet_name)
>
>        # Create a CSV file for each sheet
>        output_csv_file = f"{output_csv_folder}/{sheet_name}.csv"
>        df.to_csv(output_csv_file, index=False, header=False)
>
>        print(f"Sheet '{sheet_name}' converted to CSV: {output_csv_file}")
>
> # Example usage
> input_excel_file = 'val_2023_MixMHCpred2.2.xlsx'
> output_csv_folder = 'val_2023_MixMHCpred2.2'
> os.makedirs(output_csv_folder, exist_ok=True)
> excel_to_csv(input_excel_file, output_csv_folder)
> ```
> 
> The ten files in the folder have the following format: XXXX-L*_I.csv (where X
> is a number and L a letter) come from 
> [Gfeller et al. 2018](https://doi.org/10.4049/jimmunol.1800914). The remaining
> 11 files come with format XXXX\*LX\*.csv come from 
> [Pyke et al. 2023](https://doi.org/10.1016/j.mcpro.2023.100506).
> 
> The file has the following format:
> ```text
> Sequence,Length,MixMHCpred2.0.2,MixMHCpred2.2,NetMHCpan4.1,MHCflurry2.0,HLAthena,Ligand
> FPDTPLAL,8,2,0.0591461,0.019,0.07783,0.05,1
> PNHVEHTL,8,62,9.31695,0.695,14.27704,0.9278,1
> ...
> GKAARIQC,8,98,39.9603,68.75,99.2866,16.61503,0
> GESAWNLE,8,100,65.6608,68.333,99.2866,12.51793,0
> ...
> ```
> 
> Where the **first column is the peptide sequence**, the **second column is the
> peptide length**, the **third column is the MixMHCpred2.0.2 score**, the **fourth
> column is the MixMHCpred2.2 score**, the **fifth column is the NetMHCpan4.1 score**,
> the **sixth column is the MHCflurry2.0 score**, the **seventh column is the HLAthena
> score** and the **eighth column is the ligand label** (1 ligand, 0 non-ligand).
> 
> The peptide final score obtained by an HLA-I ligand predictors is expressed 
> as a %rank, which represents how the predicted binding of a peptide ranks
> compares with the one of random peptides from the human proteome (the smaller
> the %rank the better the binding (ie. it's in the top_rank% respect to the 
> random peptides)). 
> 
> ##### Data insights from the publication
>  In total the test set contains **78,011 HLA-I positive case ligands**. 
> 
> ⚠️ **4-fold excess of randomly selected peptides** from the human proteome were
> used as negatives to compute receiver operating curves (ROCs) and positive 
> predictive values (PPVs) 
> 
> **Warning:** It seems that the data they used to test is not monoallelic. 
> The possible HLA is not included, but I found the following information in the
> reference data publications:
> 
> | Filename    | HLA-A | HLA-A | HLA-B | HLA-B | HLA-C | HLA-C |
> |-------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
> | 3779-AMM_I  | A0201 | B3503 | B3508 |       | C0401 |       |
> | 3795-BMT_I  | A0201 | A2601 | B0702 | B3901 | C0702 | C1203 |
> | 3803-RE_I   | A0201 | A1101 | B3508 | B5101 | C0401 | C1502 |
> | 3805-RV_I   | A0206 | A0301 | B3501 | B5101 | C0401 | C1502 |
> | 3808-HMC_I  | A0201 | A2601 | B0801 | B1401 | C0701 | C0802 |
> | 3869-GA_I   | A0201 | A2402 | B1501 |       | C0304 | C0401 |
> | 3947-GA_I   | A0301 | A2402 | B1501 | B3906 | C0303 | C0702 |
> | 3971-ORA_I  | A1101 | A2501 | B4402 | B5001 | C0501 | C0602 |
> | 4001_I      | A0201 | A2601 | B5101 | B3801 | C0102 | C1203 |
> | 4037-DC_I   | A0301 | A6801 | B3501 | B4402 | C0401 | C0704 |
> | 124768B1    | A2402 | A2501 | B1801 | B3801 | C1203 | C1203 |
> | 1071227F    | A1101 | A6802 | B0801 | B5301 | C0401 | C0701 |
> | 1180402F    | A0301 | A6801 | B1803 | B3508 | C0401 | C0701 |
> | 117794A1    | A2402 | A2501 | B0702 | B1801 | C0701 | C0702 |
> | 30686B1     | A0101 | A2501 | B5101 | B5701 | C0602 | C1402 |
> | 1114162F    | A2601 | A6802 | B1402 | B2705 | C0802 | C1203 |
> | 1183384F    | A0301 | A0301 | B1805 | B3503 | C0401 | C1203 |
> | 1160324F    | A0203 | A1101 | B1502 | B4601 | C0102 | C0801 |
> | 122716A1    | A2501 | A3001 | B1302 | B2705 | C0202 | C0602 |
> | 1134036F    | A0301 | A2402 | B3502 | B4402 | C0401 | C0501 |
> | 1070865F    | A3201 | A6801 | B4001 | B5101 | C0304 | C1502 |
> 
> Source first 10 files: [Gfeller et al. 2018](https://doi.org/10.4049/jimmunol.1800914)
> [Supplementary Table II](https://journals.aai.org/jimmunol/article-supplement/106932/xlsx/ji_1800914_supplemental_table_2/)
>
> Source last 11 files: [Pyke et al. 2023](https://doi.org/10.1016/j.mcpro.2023.100506)
> [Supplementary Table 5](https://www.mcponline.org/cms/10.1016/j.mcpro.2023.100506/attachment/e29a43ab-3975-4f1c-bf9f-3db090b8aa1f/mmc6.xlsx)
> This table has been manually copied as `val_2023_MixMHCpred2.2_alleles_by_file.tsv`.
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

## T-Cell-pMHC recognition data

Data from Gfeller et al. (2023) is publicly available at the [supplementary
information](https://www.nature.com/articles/s41586-021-03819-2#Sec19) associated
with the publication.

According to the license, this data is only for academic use. Please contact the
authors of the original publication for commercial use.


Gfeller, D.; Schmidt, J.; Croce, G.; Guillaume, P.; Bobisse, S.; Genolet, R.; 
Queiroz, L.; Cesbron, J.; Racle, J.; Harari, A. Improved Predictions of Antigen 
Presentation and TCR Recognition with MixMHCpred2.2 and PRIME2.0 Reveal Potent 
SARS-CoV-2 CD8+ T-Cell Epitopes. _Cell Syst._ **2023**, 14, 72-83.e5. 
[![DOI:10.1016/j.cels.2022.12.002](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1016/j.cels.2022.12.002)


<details>

<summary><strong>Data description</strong></summary>


> #### Train data
> The training data is contained in the file `train_2023_PRIME2.0.csv`. 
> It corresponds to the Table S4 in the supplementary information, which 
> contains the List of immunogenic and non-immunogenic peptides used to train PRIME2.0.
> 
> The file has the following format:
> ```text
> Mutant,Allele,MixMHCpred,NetMHCpan,MHCflurry,HLAthena,PRIME1.0,Immunogenicity,SourceProt,Random 
> VMLQAPLFT,A0201,4.44635,2.323,1.32054,8.40544,100,0,ANKIB1,0 
> MLIVETAVM,A0201,3.50576,2.504,1.41484,8.4011,100,0,ANKIB1,0
> ...
> ```
> Where the **first column is the peptide sequence**, the **second column is the
> HLA allele**, the **third column is the MixMHCpred score**, the **fourth column
> is the NetMHCpan score**, the **fifth column is the MHCflurry score**, the **sixth
> column is the HLAthena score**, the **seventh column is the PRIME1.0 score**,
> the **eighth column is the immunogenicity label** (0 non-immunogenic, 100 immunogenic), 
> the **ninth column is the source protein** and the **tenth column specifies if the
> sequence was randomly generated**.
> 
> ##### Data insights from the publication
> They collected from 70 recent neo-antigen studies. This resulted in 596 verified 
> immunogenic neo-epitopes, as well as 6,084 non-immunogenic peptides tested 
> experimentally.
> 
> ⚠️ Most of the immunogenic and non-immunogenic peptides in those studies were 
> previously selected based on HLA-I ligand predictors and, as a result, show 
> much higher predicted binding to HLA-I compared with random peptides. 
> To correct for this bias in their data, they further included for each 
> neo-epitope 99 peptides randomly selected from the same source protein as
> additional negatives.
> 
> ##### About PRIME2.0
> PRIME2.0 is based on a neural network and uses as input features:
>
>   1. The predicted HLA-I presentation score (−log(%rank) of MixMHCpred2.2),
>   2. The amino acid frequency at positions with minimal impact on binding to HLA-I and more likely to face the TCR 
>   3. The length of the peptide
> 
> ##### Comparison with PRIME1.0 training set
> The training set of PRIME2.0 is more realistic than the PRIME1.0 training set
> in terms of predicted HLA-I binding of the negatives (i.e., broad coverage of 
> the range of %rank values without enrichment in predicted ligands). Moreover, 
> the use of neural networks can capture potential correlations between different
> input features.
> 
> We also provide this training set in the file `train_2020_PRIME1.0.csv` from 
> [Schmidt et al. (2021)](https://doi.org/10.1016/j.xcrm.2021.100194), which has
> the folllowing format:
> ```text
> Mutant,Allele,MixMHCpred,NetMHCpanEL,NetMHCpanBA,NetMHCpanBA_Kd,NetMHCstabpan,NetMHCstabpan_T12,MHCflurry,HLAthena,NetChop,TAP,IEDB,Foreignness,WT_peptide,NetMHCpanEL_WT,NetMHCpanBA_Kd_WT,ratio_%rank,ratio_Kd,DisToSelf,DisToSelf_peptide,Immunogenicity,StudyOrigin,PRIME
> VMLQAPLFT,A0201,7,2.391,2.193,356.05,8.5,0.54,0.8535,8.40544,0.03431,-0.35,-0.03346,0,DMLQAPLFT,12.165,10345.08,1.6268512877593,0.0344173268838907,9,DMLQAPLFT,0,Bobisse,0
> MLIVETAVM,A0201,7,1.936,1.61,206.86,9,0.53,1.335,8.4011,0.90141,0.301,0.27592,0,MLIVETADM,3.475,569.1,0.58497049016237,0.363486206290634,9,MLIVETADM,0,Bobisse,0
> ...
> ```

> #### Test data
> They performed multiple cross-validations based on:
> - Randomly splitting the data (standard 10-fold cross-validation), 
> - Iteratively excluding specific alleles (leave-one-allele-out cross-validation)
> - Iteratively excluding data from specific studies (leave-one-study-out cross-validation)
> - They also restricted their benchmark to peptides experimentally tested (i.e., excluding random negatives from the test sets)

</details>

## Neoantigen immunogenicity data

Data from Müller et al. (2023) is publicly available at the publication's [Data availability
section](https://www.cell.com/immunity/fulltext/S1074-7613(23)00406-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1074761323004065%3Fshowall%3Dtrue#sectitle0030).

Müller, M.; Huber, F.; Arnaud, M.; Kraemer, A. I.; Altimiras, E. R.; Michaux, J.;
Taillandier-Coindard, M.; Chiffelle, J.; Murgues, B.; Gehret, T.; Auger, A.; 
Stevenson, B. J.; Coukos, G.; Harari, A.; Bassani-Sternberg, M. 
Machine Learning Methods and Harmonized Datasets Improve Immunogenic Neoantigen 
Prediction. Immunity 2023, 56, 2650-2663.e6.
[![DOI:10.1016/j.immuni.2023.09.002](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.4049/jimmunol.1600582)

<details>

<summary><strong>Data description</strong></summary>

For memory space reasons, we provide a reduced version of the original data from
the Data section (https://figshare.com/s/147e67dde683fb769908). It contains:

> #### Train and test data
> The training and test data is contained in the files `Neopep_data_org_train.csv` and,
> `Neopep_data_org_test.csv` respectively.
> The train dataset contains data from one external study (NCI) while the test
> dataset has information from one external study (TESLA) and one in-house study from 
> the paper authors (HiTIDe):
>
> - **NCI dataset** ([Gartner et al.](https://doi.org/10.1038/s43018-021-00197-6)):
> Many mutations and neo-peptides were physically screened in a cohort of 112 
> patients. For almost all the expressed mutations, minigenes encoding the mutations
> and 12 flanking wild-type (WT) aa on each side were transcribed in vitro and 
> transfected into autologous antigens presenting cells (APCs) followed by a 
> co-culture with TIL cultures and interferon (IFN)-γ enzyme-linked immunospot
> (ELISpot) immunogenicity measurement.
> For 80 of the 112 patients, additional immunogenicity screens were performed 
> to identify the optimal neo-antigenic epitopes and their HLA restrictions. The
> top-ranked neo-peptides predicted by NetMHCpan to span immunogenic mutations 
> from the above mini-gene assay were pulsed on autologous APCs or APCs engineered 
> to express the patient’s HLA-I alleles, prior to co-culture with TILs and IFN-γ
> ELISpot readout. Neo-peptides with positive ELISpot readout were assigned as 
> immunogenic. **All other neo-peptides containing the immunogenic mutation and all 
> neo-peptides containing screened non-immunogenic mutations were considered as
> non-immunogenic.**
>   - ⚠️ **WARNING:** They only predicted immunogenicity for the best binding neo-peptides
>        according to NetMHCpan, and not for all possible neo-peptides.
> - **TESLA dataset** ([Wells et al.](https://doi.org/10.1016/j.cell.2020.09.015)): 
> It contains data for 8 patients (five with skin cutaneous melanoma, and three with NSCLC).
> The immunogenicity of selected neo-peptides was determined with labeling of 
> subject-matched TILs or peripheral blood mononuclear cells (PBMCs) with HLA-I 
> peptide multimers. 
> Müller et al. inferred annotations for the mutations from annotations of the 
> neo-peptides, where a mutation is called immunogenic when at least one of its 
> neo-peptides was reported as immunogenic, non-immunogenic when at least one of
> its neo-peptides was screened but none was found to be not immunogenic,
> and not screened otherwise.
>   - ⚠️ **WARNING:** Only a selection of neo-peptides was experimentally screened. 
>   The ones that were not tested were considered non-immonogenic.
> - **HiTIDe dataset** (from the reference paper):  
> The study included 11 patients with metastatic melanoma, lung, kidney, and stomach cancers.
> The immunogenicity of selected neo-peptides in the HiTIDe cohort was interrogated
> with IFN-γ ELISpot assays following incubation of the peptides with either bulk
> TILs or neoantigen enriched TILs (NeoScreen method) grown from tumor biopsies 
> in the presence of APCs loaded with neo-peptides. 
>   - ⚠️ **WARNING:** Only a selection of neo-peptides was experimentally screened. 
>   The ones that were not tested were considered non-immonogenic.
> 
> The file has the following format:
> ```text
> patient	dataset	train_test	response_type	Nb_Samples	Sample_Tissue	Cancer_Type	chromosome	genomic_coord	ref	alt	gene	protein_coord	aa_mutant	aa_wt	mutant_seq	wt_seq	pep_mut_start	TumorContent	mutation_type	Zygosity	mutant_best_alleles	wt_best_alleles	mutant_best_alleles_netMHCpan	mutant_other_significant_alleles_netMHCpan	wt_best_alleles_netMHCpan	mutant_rank	mutant_rank_netMHCpan	mutant_rank_PRIME	mut_Rank_Stab	TAP_score	mut_netchop_score_ct	mut_binding_score	mut_is_binding_pos	mut_aa_coeff	DAI_NetMHC	DAI_MixMHC	DAI_NetStab	mutant_other_significant_alleles	DAI_MixMHC_mbp	rnaseq_TPM	rnaseq_alt_support	GTEx_all_tissues_expression_mean	Sample_Tissue_expression_GTEx	TCGA_Cancer_expression	bestWTMatchScore_I	bestWTMatchOverlap_I	bestMutationScore_I	bestWTPeptideCount_I	bestWTMatchType_I	CSCAPE_score	Clonality	CCF	nb_same_mutation_Intogen	mutation_driver_statement_Intogen	gene_driver_Intogen	seq_len
> 4278	NCI	train	negative		Colon	Colon Adenocarcinoma	16	19718494.0	T	G	KNOP1	372.0	P	Q	DPKLKFLRL	DQKLKFLRL	2.0	0.33	SNV	HET	B0801	B0801	B0801	B1401	B0801	0.02	0.006	0.01	2.5	0.048	0.93382	-0.0088407261858941	True	-1.5	-0.9162907318741551	0.0	0.5108256237659906	1	0.0	34.6839	24.14	3.628448275862069	3.38	17.748463778707407	9.0	1.0	1.0	26	EXACT	0.708244	clonal	0.8768072885719944				9
> ...
> ```
> Where the **first column is the patient ID**, the **second column is the dataset**,
> the **third column is the train/test split** and the **fourth column is the response 
> type (ie. immunogenicity)** (CD8 (ie. positive)/negative/not_tested).
> 
> Patient HLAs are found in `HLA_allotypes.txt` file.
</details>