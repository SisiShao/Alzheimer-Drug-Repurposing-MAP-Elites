import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
SEEDS = [1, 2, 3, 4, 5]
METRICS = ['roc_auc_score', 'pr_auc', 'f1_score', 'precision', 'recall']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "../data/processed_data")


def get_stats_string(data_list):
    if not data_list: return "N/A"
    mean = np.mean(data_list)
    std = np.std(data_list)
    return f"{mean:.3f} ± {std:.3f}"


def main():
    print(f"\n--- Processing XGBoost Statistics ---")

    stats_storage = {m: [] for m in METRICS}
    successful_seeds = 0

    for seed in SEEDS:
        filename = os.path.join(RESULTS_DIR, f"result_xgboost_seed{seed}.csv")

        if not os.path.exists(filename):
            # Fallback
            filename_local = os.path.join(CURRENT_DIR, f"../2_automl/result_xgboost_seed{seed}.csv")
            if os.path.exists(filename_local):
                filename = filename_local
            else:
                continue

        df = pd.read_csv(filename)
        successful_seeds += 1

        for m in METRICS:
            stats_storage[m].append(df[m].iloc[0])

    if successful_seeds == 0:
        print("No XGBoost data found.")
        return

    print(f"  > Aggregated {successful_seeds} seeds.")

    row = {"Method": "XGBoost (Baseline)", "N_Seeds": successful_seeds}
    for m in METRICS:
        row[m] = get_stats_string(stats_storage[m])

    final_df = pd.DataFrame([row])
    cols = ['Method', 'N_Seeds'] + METRICS
    final_df = final_df[cols]

    print("\n=== XGBoost Summary Statistics ===")
    print(final_df)

    outfile = "summary_stats_xgboost.csv"
    final_df.to_csv(outfile, index=False)
    print(f"\nSaved table to {outfile}")


if __name__ == "__main__":
    main()