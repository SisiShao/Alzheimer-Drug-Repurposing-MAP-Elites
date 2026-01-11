import pandas as pd
import numpy as np
import argparse
import os
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Setup Paths ---
# Use relative path logic
current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    SEED = args.seed
    print(f"Starting XGBoost Baseline Run (Seed={SEED})...")

    # 1. File Paths
    # Assumption: data is in ../data/processed_data relative to this script
    data_dir = os.path.join(current_dir, "../data/processed_data")
    train_file = os.path.join(data_dir, "final_X_train.csv")

    # --- SAFETY CHECK ---
    if not os.path.exists(train_file):
        print("\n" + "="*60)
        print("DATA NOT FOUND")
        print(f"Could not find input data at: {train_file}")
        print("Please refer to data/README.md for instructions on accessing")
        print("the NIAGADS dataset and preparing the input files.")
        print("="*60 + "\n")
        return
    # --------------------

    # 2. Load Data
    # Note: XGBoost uses raw features; it doesn't need the embedding distances.
    print("Loading data...")
    X_train = pd.read_csv(train_file)
    y_train = np.array(pd.read_csv(os.path.join(data_dir, "final_y_train.csv")).iloc[:, -1])
    X_test = pd.read_csv(os.path.join(data_dir, "final_X_test.csv"))
    y_test = np.array(pd.read_csv(os.path.join(data_dir, "final_y_test.csv")).iloc[:, -1])

    # 3. Define Pipeline (Scaling + XGBoost)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            n_jobs=16,           # Parallel threads
            random_state=SEED,
            eval_metric='logloss'
        ))
    ])

    # 4. Define Hyperparameter Search Space
    # A robust search space proves rigorous comparison
    param_dist = {
        'xgb__n_estimators': [100, 200, 300, 500],
        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgb__max_depth': [3, 5, 7, 10],
        'xgb__min_child_weight': [1, 3, 5],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    }

    # 5. Run Randomized Search (CV)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,              # Budget: 20 combinations
        scoring='roc_auc',      # Optimize AUC
        cv=5,
        verbose=1,
        random_state=SEED,
        n_jobs=1                # Let XGBoost handle parallelism
    )

    print("Tuning hyperparameters (5-fold CV)...")
    search.fit(X_train, y_train)
    
    print(f"Best Params: {search.best_params_}")
    best_model = search.best_estimator_

    # 6. Evaluate on Test Set
    print("Evaluating best model...")
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred_class = best_model.predict(X_test)

    # Calculate Metrics
    results = {
        'seed': SEED,
        'roc_auc_score': roc_auc_score(y_test, y_pred_prob),
        'pr_auc': average_precision_score(y_test, y_pred_prob),
        'f1_score': f1_score(y_test, y_pred_class),
        'precision': precision_score(y_test, y_pred_class),
        'recall': recall_score(y_test, y_pred_class)
    }

    # 7. Save Results
    df = pd.DataFrame([results])
    output_filename = f"result_xgboost_seed{SEED}.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"DONE! Results:\n{df}")
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    main()