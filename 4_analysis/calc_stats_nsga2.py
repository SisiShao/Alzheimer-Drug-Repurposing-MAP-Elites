import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
DIMS = [16, 32]
SEEDS = [1, 2, 3, 4, 5]
METRICS = ['roc_auc_score', 'pr_auc', 'f1_score', 'precision', 'recall']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "../data/processed_data")


def get_stats_string(data_list):
    if not data_list: return "N/A"
    mean = np.mean(data_list)
    std = np.std(data_list)
    return f"{mean:.3f} ± {std:.3f}"


def analyze_dimension(dim):
    print(f"\n--- Processing NSGA-II Dimension {dim} ---")

    stats_storage = {
        "Best Performance": {m: [] for m in METRICS},
        "Most Novel": {m: [] for m in METRICS}
    }

    successful_seeds = 0

    for seed in SEEDS:
        filename = os.path.join(RESULTS_DIR, f"result_nsga2_dim{dim}_seed{seed}.csv")

        if not os.path.exists(filename):
            # Fallback to local run
            filename_local = os.path.join(CURRENT_DIR, f"../2_automl/result_nsga2_dim{dim}_seed{seed}.csv")
            if os.path.exists(filename_local):
                filename = filename_local
            else:
                continue

        df = pd.read_csv(filename)
        successful_seeds += 1

        # Best Performance
        best_idx = df['roc_auc_score'].idxmax()
        best_row = df.loc[best_idx]
        for m in METRICS:
            stats_storage["Best Performance"][m].append(best_row[m])

        # Most Novel
        df['coord_sum'] = df['pipeline_scores_Euc'] + df['pipeline_scores_Canberra']
        max_dist_in_run = df['coord_sum'].max()
        furthest_candidates = df[df['coord_sum'] == max_dist_in_run]
        novel_idx = furthest_candidates['roc_auc_score'].idxmax()
        novel_row = furthest_candidates.loc[novel_idx]
        for m in METRICS:
            stats_storage["Most Novel"][m].append(novel_row[m])

    if successful_seeds == 0:
        print(f"  [Info] No data found for Dim {dim}")
        return None

    print(f"  > Aggregated {successful_seeds} seeds.")

    rows = []
    for strategy in ["Best Performance", "Most Novel"]:
        row = {"Dimension": dim, "Strategy": strategy, "N_Seeds": successful_seeds}
        for m in METRICS:
            row[m] = get_stats_string(stats_storage[strategy][m])
        rows.append(row)

    return pd.DataFrame(rows)


# --- MAIN ---
all_stats = []
for dim in DIMS:
    df_dim = analyze_dimension(dim)
    if df_dim is not None:
        all_stats.append(df_dim)

if all_stats:
    final_table = pd.concat(all_stats, ignore_index=True)
    cols = ['Dimension', 'Strategy', 'N_Seeds'] + METRICS
    final_table = final_table[cols]

    print("\n=== NSGA-II Summary Statistics ===")
    print(final_table)

    outfile = "summary_stats_nsga2.csv"
    final_table.to_csv(outfile, index=False)
else:
    print("\nNo NSGA-II data found.")