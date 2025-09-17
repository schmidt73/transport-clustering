import argparse
import os
import json

import sklearn.metrics as metrics
import numpy as np
import pandas as pd

SIM_DATA = "/n/fs/ragr-research/projects/convex_lrot/simulated_data"

def process_result(setting, instance, result_path):
    summary_file = f"{result_path}/result_summary.json"
    Q_matrix     = f"{result_path}/result_Q.txt"
    R_matrix     = f"{result_path}/result_R.txt"
    X_clusters   = f"{SIM_DATA}/{setting}/{instance}_labels_X.txt"
    Y_clusters   = f"{SIM_DATA}/{setting}/{instance}_labels_Y.txt"

    Q, R = np.loadtxt(Q_matrix), np.loadtxt(R_matrix)
    Q_clusters = np.argmax(Q, axis=1)
    R_clusters = np.argmax(R, axis=1)
    X_clusters, Y_clusters = np.loadtxt(X_clusters), np.loadtxt(Y_clusters)

    print(setting, instance, result_path)
    Q_ari = metrics.adjusted_rand_score(X_clusters, Q_clusters)
    R_ari = metrics.adjusted_rand_score(Y_clusters, R_clusters)
    Q_ami = metrics.adjusted_mutual_info_score(X_clusters, Q_clusters)
    R_ami = metrics.adjusted_mutual_info_score(X_clusters, R_clusters)

    with open(summary_file, "r") as f:
        row = json.load(f) 

    del row['lower_bound']
    del row['l1_row_marginal_error']
    del row['l1_col_marginal_error']
    del row['l1_total_error']
    del row['simulation_seed']

    row['X_ari'] = Q_ari
    row['Y_ari'] = R_ari
    row['X_ami'] = Q_ami
    row['Y_ami'] = R_ami

    return row

def summarize_results(root_directory):
    rows = []
    for subdir in os.listdir(root_directory):
        if 'clique' in subdir: continue
        alg, rank, seed, setting = subdir.split("_", 3)
        for instance in os.listdir(root_directory + "/" + subdir):
            result_path = root_directory + "/" + subdir + "/" + instance
            row = process_result(setting, instance, result_path)
            row['algorithm'] = alg
            row['rank'] = rank
            row['seed'] = seed
            row['setting'] = setting
            row['instance'] = instance
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="nextflow_results/algorithms",
                    help="Root directory containing algorithm result folders.")
    ap.add_argument("--out", type=str, default="nextflow_results/algorithms_summary.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    df = summarize_results(args.root)
    if df.empty:
        print(f"[INFO] No results found under: {args.root}")
        return

    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
