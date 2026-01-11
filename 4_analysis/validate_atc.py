import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import sys

# --- Configuration ---
# Allow relative paths to work regardless of where the script is run
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "../data/processed_data")

# Adjust filenames to match what you actually upload (e.g., dim32 or dim16)
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings_drug_vgae_ADcluster.csv")
MAPPING_FILE = os.path.join(DATA_DIR, "drug_atc_mapping.csv")
OUTPUT_PLOT = "atc_validation_boxplot.png"
DPI = 300


def main():
    print("--- Starting ATC validation (Mann-Whitney U) ---")

    # 1. Read embedding file
    if not os.path.exists(EMBEDDING_FILE):
        print(f"Error: Cannot find embedding file at: {EMBEDDING_FILE}")
        print("Please ensure you have uploaded the trained embeddings to 'data/processed_data/'.")
        return

    df = pd.read_csv(EMBEDDING_FILE)

    # Auto-detect columns: Assumes Col 1 = NodeID, Col 2 = Name, Col 3+ = Vector
    # (Adjust this logic if your CSV structure is different)
    name_col = df.columns[1]
    emb_cols = [c for c in df.columns if c.startswith('embedding_') or c.startswith('dim_')]

    if not emb_cols:
        # Fallback to column index if names don't match
        emb_cols = df.columns[2:]

    # Clean drug names
    df["clean_name"] = df[name_col].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=["clean_name"])
    df_emb = df.set_index("clean_name")[emb_cols]

    print(f"Loaded embeddings for {len(df_emb)} drugs.")

    # 2. Read mapping file
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: Cannot find ATC mapping file at: {MAPPING_FILE}")
        print("Please upload 'drug_atc_mapping.csv' to 'data/processed_data/'.")
        return

    df_map = pd.read_csv(MAPPING_FILE)
    # Flexible column reading for mapping file
    if len(df_map.columns) >= 2:
        df_map.columns = ["drug_name", "atc_code"]  # Force rename for simplicity
        df_map["clean_name"] = df_map["drug_name"].astype(str).str.upper().str.strip()
        df_map["atc_level1"] = df_map["atc_code"].astype(str).str[0]
        atc_dict = pd.Series(df_map.atc_level1.values, index=df_map.clean_name).to_dict()
    else:
        print("Error: Mapping file format incorrect. Need at least 2 columns (Name, Code).")
        return

    # 3. Match drugs
    matched_drugs = [d for d in df_emb.index if d in atc_dict]
    print(f"Matched {len(matched_drugs)} drugs with ATC codes.")

    if len(matched_drugs) < 10:
        print("Too few matched drugs to perform statistical analysis.")
        return

    # 4. Compute distances
    intra_dist = []
    inter_dist = []

    # Downsample if too many to save time (optional)
    if len(matched_drugs) > 500:
        print("   (Sampling 500 drugs for pairwise comparisons to speed up calculation...)")
        matched_drugs = np.random.choice(matched_drugs, 500, replace=False)

    pairs = list(itertools.combinations(matched_drugs, 2))
    print(f"Computing distances for {len(pairs)} pairs...")

    for d1, d2 in pairs:
        vec1 = df_emb.loc[d1].values.astype(float)
        vec2 = df_emb.loc[d2].values.astype(float)
        d = dist.euclidean(vec1, vec2)

        if atc_dict[d1] == atc_dict[d2]:
            intra_dist.append(d)
        else:
            inter_dist.append(d)

    # 5. Statistics
    mean_intra = np.mean(intra_dist) if len(intra_dist) else 0
    mean_inter = np.mean(inter_dist) if len(inter_dist) else 0

    print("\n=== Validation Results ===")
    print(f"Mean distance (Same ATC Class):      {mean_intra:.4f}")
    print(f"Mean distance (Different ATC Class): {mean_inter:.4f}")

    if len(intra_dist) >= 2 and len(inter_dist) >= 2:
        u_stat, p_val = mannwhitneyu(intra_dist, inter_dist, alternative='less')
        print(f"Mann-Whitney U p-value:          {p_val:.4e}")

        if p_val < 0.05:
            print("RESULT: Significant! Drugs in the same class are clustered closer together.")
        else:
            print("result: Not significant.")
    else:
        print("Insufficient data for statistics.")

    # 6. Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 6), dpi=DPI)

    plot_data = pd.DataFrame({
        "Distance": intra_dist + inter_dist,
        "Group": ["Same ATC Class"] * len(intra_dist) + ["Diff ATC Class"] * len(inter_dist)
    })

    pastel_palette = sns.color_palette("Pastel1", 2)

    ax = sns.boxplot(
        x="Group", y="Distance", data=plot_data,
        palette=pastel_palette, showfliers=False, width=0.55, linewidth=1.2
    )

    plt.title("Biological Validity Check (ATC Classification)")
    plt.ylabel("Euclidean Distance in Embedding Space")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.savefig(OUTPUT_PLOT, dpi=DPI, bbox_inches="tight")
    print(f"Plot saved as {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()