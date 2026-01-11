import os
import sys
import warnings
import numpy as np
import pandas as pd
import json
import argparse
from sklearn import *
import sklearn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import partial

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check for TPOT2 library
try:
    import tpot2
except ImportError:
    print("\n[WARNING] 'tpot2' library not found.")
    print("This script requires the TPOT2 library to run.")
    tpot2 = None

# --- HELPER FUNCTIONS ---
def discretize_scores(scores):
    discretized_scores = np.zeros(scores.shape)
    for i in range(scores.shape[1]):
        quantiles = np.quantile(scores[:, i], [0.25, 0.5, 0.75])
        discretized_scores[:, i] = np.digitize(scores[:, i], bins=quantiles, right=True)
    return discretized_scores

# --- MAIN EXECUTION ---
def main():
    if tpot2 is None:
        return

    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dim", type=int, default=16, help="Embedding dimension (16 or 32)")
    args = parser.parse_args()
    
    SEED = args.seed
    DIM = args.dim
    
    print(f"Starting NSGA-II (Standard) Baseline: Dimension={DIM}, Seed={SEED}")
    warnings.filterwarnings("ignore")
    
    # 2. File Paths
    # Relative paths assuming structure: 2_automl/script.py and data/processed_data/
    data_dir = os.path.join(current_dir, "../data/processed_data")
    
    score_file = os.path.join(data_dir, f"feature_sets_scores_dim{DIM}.csv")
    json_file = os.path.join(data_dir, f"feature_sets_ensemble_dim{DIM}.json")
    
    train_file = os.path.join(data_dir, "final_X_train.csv")
    
    # --- SAFETY CHECK ---
    if not os.path.exists(train_file):
        print("\n" + "="*60)
        print(" DATA NOT FOUND")
        print(f"Could not find input data at: {train_file}")
        print("Please refer to data/README.md for instructions.")
        print("="*60 + "\n")
        return
    # --------------------
    
    # Load Data
    X_train = pd.read_csv(train_file)
    y_train = np.array(pd.read_csv(os.path.join(data_dir, "final_y_train.csv")).iloc[:, -1])
    X_test = pd.read_csv(os.path.join(data_dir, "final_X_test.csv"))
    y_test = np.array(pd.read_csv(os.path.join(data_dir, "final_y_test.csv")).iloc[:, -1])
    
    # Load Scores
    scores_df = pd.read_csv(score_file).iloc[:, :-1].dropna()
    with open(json_file, 'r') as f:
        fss_list = json.load(f)
        
    scores_arr = np.array(scores_df)
    discretized_scores = discretize_scores(scores_arr)

    # TPOT Config
    fss_search_space = tpot2.search_spaces.nodes.FSSNode(subsets=fss_list)
    s_union_c_sp2_pass = tpot2.search_spaces.pipelines.SequentialPipeline(search_spaces=[
        tpot2.config.get_search_space(["scalers", "Passthrough"]),
        tpot2.config.get_search_space(["selectors", "selectors_classification", "Passthrough"]),
        tpot2.search_spaces.pipelines.ChoicePipeline([
            tpot2.search_spaces.pipelines.DynamicUnionPipeline(tpot2.config.get_search_space(["transformers"])),
            tpot2.config.get_search_space("SkipTransformer"),
            tpot2.config.get_search_space("Passthrough")
        ]),
        tpot2.config.get_search_space(["classifiers"]),
    ])
    combined_search_space = tpot2.search_spaces.pipelines.SequentialPipeline([fss_search_space, s_union_c_sp2_pass])

    # 3. Initialize TPOT (STANDARD NSGA-II MODE)
    # CRITICAL: No survival_selector, No parent_selector passed here
    est = tpot2.TPOTEstimator(
        scorers=["roc_auc"],
        scorers_weights=[1],
        classification=True,
        cv=5,
        search_space=combined_search_space,
        population_size=100,
        generations=100,
        max_eval_time_seconds=60 * 10,
        max_time_seconds=None,
        verbose=1,
        n_jobs=16,
        memory_limit="60GB",
        random_state=SEED 
    )

    est.fit(X_train, y_train)

    # 4. Evaluation
    final_pop_df = est._evolver_instance.population.evaluated_individuals.loc[
        [e.unique_id() for e in est._evolver_instance.population.population]
    ]
    
    # Initialize columns
    cols = ['pipeline_scores_Euc', 'pipeline_scores_Canberra', 'pipeline_scores_cosine', 
            'selected_fs', 'roc_auc_score', 'pr_auc', 'f1_score', 'precision', 'recall']
    for c in cols:
        final_pop_df[c] = 0.0

    print("Evaluating NSGA-II pipelines...")
    for i in range(final_pop_df.shape[0]):
        try:
            sklearn_pipeline = final_pop_df['Individual'][i].export_pipeline()
            sklearn_pipeline.fit(X_train, y_train)
            
            y_pred_prob = sklearn_pipeline.predict_proba(X_test)[:, 1]
            y_pred_class = sklearn_pipeline.predict(X_test)
            
            # --- Calculate ALL Metrics ---
            roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
            pr_auc = sklearn.metrics.average_precision_score(y_test, y_pred_prob)
            f1 = sklearn.metrics.f1_score(y_test, y_pred_class)
            precision = sklearn.metrics.precision_score(y_test, y_pred_class)
            recall = sklearn.metrics.recall_score(y_test, y_pred_class)
            
            first_step = int(sklearn_pipeline.steps[0][1].name)
            
            # --- Save Metrics & Coordinates ---
            idx_loc = final_pop_df.index.get_loc(final_pop_df.index[i])
            
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('pipeline_scores_Euc')] = discretized_scores[first_step, 0]
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('pipeline_scores_Canberra')] = discretized_scores[first_step, 1]
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('pipeline_scores_cosine')] = discretized_scores[first_step, 2]
            
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('roc_auc_score')] = roc_auc
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('selected_fs')] = first_step
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('pr_auc')] = pr_auc
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('f1_score')] = f1
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('precision')] = precision
            final_pop_df.iloc[idx_loc, final_pop_df.columns.get_loc('recall')] = recall
            
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue

    # 5. Save CSV Results
    output_filename = f"result_nsga2_dim{DIM}_seed{SEED}.csv"
    final_pop_df.to_csv(output_filename, index=True)
    print(f"DONE! Saved results to {output_filename}")

    # 6. Plotting (Embedded)
    try:
        print("Generating NSGA-II Plot...")
        grouped = final_pop_df.groupby(['pipeline_scores_Euc', 'pipeline_scores_Canberra'])
        result = grouped.apply(lambda x: x.loc[x['roc_auc_score'].idxmax()])
        result = result.reset_index(drop=True)
        result = result[['pipeline_scores_Euc', 'pipeline_scores_Canberra', 'selected_fs', 'roc_auc_score']]
        
        selected_scores = np.array(result.iloc[:, :2])
        max_scores = np.array(result.iloc[:, -1])
        selected_fs = result['selected_fs']
        x_ticks = np.unique(selected_scores[:, 0])
        y_ticks = np.unique(selected_scores[:, 1])
        levels = np.linspace(min(max_scores), max(max_scores), 8)
        colors = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
        
        fig, ax = plt.subplots()
        for i in range(len(selected_scores)):
            x = selected_scores[i, 0]
            y = selected_scores[i, 1]
            color_value = max_scores[i]
            ax.fill_between([x-0.5, x+0.5], y-0.5, y+0.5, color=cmap(norm(color_value)))
            ax.text(x, y, f'{int(list(selected_fs)[i])}', ha='center', va='center', fontsize=10, color='black')
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_aspect('equal', adjustable='box')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        char = fig.colorbar(sm, orientation='vertical', ax=ax, ticks=levels)
        char.set_label('ROC_AUC of the pipeline')
        
        # NOTE: Plot filename includes 'nsga2'
        plot_filename = f"plot_nsga2_dim{DIM}_seed{SEED}.png"
        plt.title(f"NSGA-II Grid (Dim={DIM}, Seed={SEED})")
        plt.xlabel("Euclidean")
        plt.ylabel("Canberra")
        
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")
        
    except Exception as e:
        print(f"Plotting failed (CSV is safe): {e}")

if __name__ == "__main__":
    main()