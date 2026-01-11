# Alzheimer-Drug-Repurposing-MAP-Elites

This repository contains the code and processed artifacts associated with our study:

**“A Biology-Based Quality-Diversity Algorithm for Drug Repurposing in Alzheimer’s Disease Using Automated Machine Learning.”**

The goal of this work is to identify *mechanistically novel* drug repurposing candidates for Alzheimer’s disease (AD) by combining knowledge-graph-derived node embeddings with a diversity-driven AutoML search.

---

## Project Overview

Drug repurposing for neurodegenerative diseases is challenging: traditional machine-learning pipelines often optimise solely for predictive accuracy and tend to converge on well-characterised targets. Our approach integrates:

### Graph neural network (GNN) embeddings
We use **GraphSAGE** to embed genes and a **Variational Graph Autoencoder (VGAE)** to embed drugs based on their connectivity in the Alzheimer’s Knowledge Base (AlzKB).  
A clustering loss encourages known AD genes and drugs to cluster tightly in latent space, providing a biologically grounded reference for defining novelty.

### Feature set construction
GNN embeddings are converted into numeric distances and clustered via **K-means** to create biologically informed feature sets.  
A small number of high-degree “hub” drugs are isolated into an outlier group to preserve mechanistic specificity.

### Automated model search
We extend the **TPOT2 AutoML** system by integrating the **MAP-Elites quality–diversity algorithm**.  
This grid-based evolutionary search explores a wide variety of pipelines (e.g. tree-based models, ensembles) across multiple novelty dimensions defined by the embeddings, encouraging discovery of high-performing yet mechanistically distinct solutions.

### Baselines for comparison
Scripts are provided to run:
- standard **XGBoost** pipelines, and  
- a **TPOT + NSGA-II** multi-objective search,  

allowing direct performance comparisons with MAP-Elites.

---

## Repository Structure  

```text
.
├── 1_embeddings/               # Scripts to train GNN embeddings for genes and drugs
│   ├── example_train_genes.py      # GraphSAGE example for gene embeddings
│   └── example_train_drugs.py      # VGAE example for drug embeddings
├── 2_TPOT2+MapElites/          # Scripts to run MAP-Elites and baseline AutoML methods
│   ├── run_MAP_Elites_Seed.py     # Main MAP-Elites search driver (with seed control)
│   ├── run_TPOT_NSGA2.py          # Baseline search using NSGA-II
│   └── run_XGBoost.py             # Baseline tree-based model (XGBoost)
├── 3_visualization/            # Visualisation utilities
│   └── plot_figure1_umap.py       # Generates UMAP plots of GNN embeddings
├── 4_analysis/                 # Statistical analysis and validation scripts
│   ├── calc_stats_map_elites.py   # Computes mean ± SD metrics for MAP-Elites runs
│   ├── calc_stats_xgboost.py      # Computes statistics for XGBoost runs
│   ├── calc_stats_nsga2.py        # Computes statistics for NSGA-II runs
│   └── validate_atc.py            # Validates drug embeddings against ATC categories
└── data/                       # Processed data and availability statement
    ├── DATA_AVAILABILITY.md       # Data directory & privacy statement (AlzKB/NIAGADS)
    └── processed_data/ …          # Embeddings, feature sets, sample results, etc.
```

---

## Directory Details

### `1_embeddings/`

The scripts `example_train_genes.py` and `example_train_drugs.py` demonstrate how to:

- connect to **AlzKB** (Neo4j or Memgraph),
- extract relevant subgraphs, and
- train node embeddings using **GraphSAGE** (genes) and **VGAE** (drugs).

The outputs are CSV files containing node identifiers and learned embedding coordinates.

> **Note:**  
> The provided scripts use identity (one-hot) node features for demonstration.  
> In production use, richer biological features (e.g. gene expression, chemical descriptors) can be incorporated.

---

### `2_TPOT2+MapElites/`

This directory contains the evolutionary search engines used in the study:

- **`run_MAP_Elites_Seed.py`**  
  Executes the MAP-Elites–enhanced TPOT2 search, evolving pipelines over multiple generations and producing an archive of high-performing yet diverse models.  This implementation is intended as a computational proof-of-concept rather than a clinically deployable system.

  The script accepts a random seed and embedding dimension as arguments.

- **`run_TPOT_NSGA2.py`**  
  Runs a multi-objective TPOT search using the NSGA-II algorithm as a strong baseline.

- **`run_XGBoost.py`**  
  Fits an XGBoost classifier on the same feature sets for comparison.

Outputs from these scripts can be summarised using the analysis utilities in `4_analysis/`.

---

### `3_visualization/`

- **`plot_figure1_umap.py`**  
  Reads pre-trained embeddings and applies **UMAP** to project them into two dimensions for inspection.  
  Known AD genes/drugs are highlighted, while coloured clusters illustrate embedding topology.

---

### `4_analysis/`

These scripts aggregate results across multiple random seeds and configurations:

- **`calc_stats_map_elites.py`**, **`calc_stats_xgboost.py`**, **`calc_stats_nsga2.py`**  
  Compute mean ± standard deviation for performance metrics (ROC-AUC, PR-AUC, F1-score, precision, recall).

- **`validate_atc.py`**  
  Verifies that drugs with similar embeddings share **Anatomical Therapeutic Chemical (ATC)** classes, providing biological validation.

---

## Data Availability

Raw clinical and genetic data from the **Alzheimer's Disease Sequencing Project (ADSP)** are subject to access restrictions and are **not included** in this repository.

Instructions for requesting authorised access are provided in:
 `data/DATA_AVAILABILITY.md`

The `data/processed_data/` directory contains intermediate results—such as embeddings, clustered feature sets, ATC mappings, and example pipeline outputs—which allow reproduction of analyses and figures without access to protected raw data.

---

## Requirements & Installation

This project was developed with **Python 3.9**.

Key dependencies include:

- PyTorch & PyTorch Geometric  
- Neo4j Python driver or Memgraph client  
- pandas, numpy, matplotlib  
- scikit-learn, XGBoost  
- Optuna (optional, for tuning)

Example installation:

```bash
pip install torch torch_geometric pandas numpy matplotlib \
            scikit-learn xgboost neo4j optuna
```
            
For **TPOT2** and **MAP-Elites**, a development version of TPOT2 **may be required depending on the execution environment**.  
Please consult the TPOT2 documentation for installation details.
---

## Usage

### 1. Train embeddings

Edit `1_embeddings/example_train_genes.py` and `1_embeddings/example_train_drugs.py` to specify your  
Neo4j/Memgraph connection and AD gene/drug lists.

Run the scripts to generate embedding CSV files.

### 2. Prepare feature sets

Use the embeddings to construct clustered feature sets (as described in the manuscript).

Example precomputed files are provided in `data/processed_data/`.

### 3. Run AutoML pipelines

```bash
# MAP-Elites
python 2_TPOT2+MapElites/run_MAP_Elites_Seed.py --seed <seed> --dim <dim>

# NSGA-II baseline
python 2_TPOT2+MapElites/run_TPOT_NSGA2.py --seed <seed> --dim <dim>

# XGBoost baseline
python 2_TPOT2+MapElites/run_XGBoost.py --dim <dim>
```

### 4. Visualize Embeddings
```bash
python 3_visualization/plot_figure1_umap.py
```

### 5. Analyze Results
```bash
python 4_analysis/calc_stats_map_elites.py
python 4_analysis/validate_atc.py
```

---
## Known Issues & Tips

- **Graph backend**  
  Example scripts use Neo4j (Bolt protocol). Adjust connection settings if using Memgraph.

- **Node features**  
  Identity features are used for demonstration. Adding biologically meaningful features may improve performance.

- **Environment files**  
  No `requirements.txt` or `environment.yml` is provided.  
  Record package versions when reproducing experiments.

- **License**  
  No license file is currently included. Please contact the authors if you intend to reuse or extend the code.
---

## Citation

If you use this repository, its methodology, or derived software artifacts, please cite:

> Shao S., Ribeiro P. H., Orlenko A., *et al.*  
> **A Biology-Based Quality-Diversity Algorithm for Drug Repurposing in Alzheimer’s Disease Using Automated Machine Learning.**  
> *Manuscript under review.*

A preprint or published version will be linked here upon acceptance.

```bibtex
@unpublished{shao2026biodiversity,
  title   = {A Biology-Based Quality-Diversity Algorithm for Drug Repurposing in Alzheimer’s Disease Using Automated Machine Learning},
  author  = {Shao, Sisi and Ribeiro, Pedro H. and Orlenko, Alexander and others},
  note    = {Manuscript under review},
  year    = {2026}
}
```

