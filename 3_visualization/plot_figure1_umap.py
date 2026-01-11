#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: plot_figure1_umap.py
Description:
    Generates Figure 1 (UMAP Visualization of Gene Embeddings).

    [VISUALIZATION NOTE]
    For demonstration purposes, this script highlights a curated subset of
    well-known Alzheimer's Disease risk genes (e.g., APOE, APP, TREM2).
    In the full manuscript analysis, the complete set of AD-associated genes
    (as defined in the training phase) was used for topological validation.

Dependencies:
    - ../1_embeddings/embeddings_gene_sage_ADcluster.csv

Author: Sisi Shao
"""

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = "../1_embeddings/embeddings_gene_sage_ADcluster.csv"
OUTPUT_IMG = "figure1_umap.png"

# Curated list of top AD risk genes for visualization demo
# (Canonical markers used to verify clustering quality)
DEMO_AD_GENES = {
    'APP', 'PSEN1', 'PSEN2', 'APOE', 'ADAM10', 'BACE1', 'MAPT',
    'TREM2', 'SORL1', 'ABCA7', 'BIN1', 'CD33', 'CLU', 'CR1', 'PICALM'
}


def main():
    print("--- Generating Figure 1 (UMAP Visualization) ---")

    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] File not found: {INPUT_CSV}")
        print("Please run the training script in '1_embeddings/' first.")
        return

    print(f"Loading embeddings from: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Check for gene symbols
    if 'geneSymbol' not in df.columns:
        print("[Error] 'geneSymbol' column missing in CSV. Cannot highlight genes by name.")
        return

    # Extract features
    embedding_cols = [col for col in df.columns if col.startswith('embedding_') or col.startswith('dim_')]
    embeddings = df[embedding_cols].values
    gene_symbols = df['geneSymbol'].values

    print(f"Loaded {len(df)} genes. Running UMAP...")

    # 2. Run UMAP
    # Parameters tuned for topological preservation
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # 3. Plotting
    plt.figure(figsize=(10, 8), dpi=300)

    # Background: All genes (Grey)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                c='lightgrey', s=15, alpha=0.5, label='Background Genes')

    # Highlight: Demo AD Genes (Red)
    # We map symbols to indices
    mask = [str(g) in DEMO_AD_GENES for g in gene_symbols]

    if any(mask):
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c='#d62728', s=80, alpha=1.0, edgecolors='white', linewidth=0.8,
                    label='Canonical AD Genes')
        print(f"Highlighted {sum(mask)} canonical AD genes present in the dataset.")
    else:
        print("Warning: None of the demo AD genes were found in the CSV file.")

    # Formatting
    plt.title("Figure 1: Gene Embedding Space (UMAP)\nDemonstrating Disease Module Clustering", fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_IMG)
    print(f" Figure saved to: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()