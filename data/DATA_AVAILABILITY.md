# Data Directory and Availability Statement

This directory contains the metadata and processed intermediate files required to reproduce the analyses presented in the manuscript.

## 🔒 Data Privacy & Access (ADSP/NIAGADS)

The clinical and genetic patient data used in this study were obtained from the **Alzheimer's Disease Sequencing Project (ADSP)**. Due to strict data usage agreements and HIPAA privacy regulations, **raw patient-level data (e.g., individual genotypes, clinical phenotypes) cannot be publicly shared in this repository.**

### How to Request Raw Data
Researchers interested in accessing the original dataset for full reproduction can apply for authorized access through **NIAGADS** (National Institute on Aging Genetics of Alzheimer's Disease Data Storage Site):

- **Dataset Accession**: [NG00067 (Version 18)]
- **Application Link**: [https://dss.niagads.org/](https://dss.niagads.org/)

---

## Directory Structure

To This allows for algorithmic reproducibility without compromising privacy, we provide the following processed files:

### 1. `processed_data/`
ThiThis folder contains the "Intermediate Results" which serve as inputs for the AutoML search and statistical validation scripts.
- `embeddings_gene_sage_ADcluster.csv`: Pre-trained topological gene embeddings.
- `embeddings_drug_vgae_ADcluster.csv`: Pre-trained drug embeddings via VGAE.
- `selected_feature_sets_threshold20_dim16.csv`: The clustered gene feature sets used in the TPOT search space.
- `drug_atc_mapping.csv`: Anatomical Therapeutic Chemical (ATC) codes used for biological validation.
- `result_dim16_seed5.csv`: A sample output from a MAP-Elites execution for immediate visualization.

### 2. `gene_lists/`
Publicly available biological knowledge used to guide the GNN training.
- `newADgenes.csv`: This file lists the genes known to be associated with Alzheimer’s disease, typically curated from literature and specialized AD databases. Each row contains a unique gene ID and its symbol.
- `pathway_genes.csv`: Genes involved in key biological pathways.
- `protein_coding_druggable_genes.csv`: Filtered list of druggable gene targets.

---

## Reproduction Workflow (Inference/Analysis Mode)

Even without access to the raw NIAGADS data, you can still verify the study's core claims using the provided processed data:

1. **Topological Analysis**: Run `3_visualization/plot_figure1_umap.py` to see how AD risk genes cluster in the learned embedding space.
2. **Biological Validation**: Run `4_analysis/validate_atc.py` to verify that drug embeddings capture therapeutic similarities.
3. **Statistical Aggregation**: Run `4_analysis/calc_stats_map_elites.py` to generate the Mean ± SD tables reported in the paper.

---

## Contact  

For questions about the code, please contact the first author listed in the main manuscript.