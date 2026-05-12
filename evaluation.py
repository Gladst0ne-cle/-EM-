import csv
import json
import os
from statistics import mean, stdev

import numpy as np
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score

from em_gmm import clustering_accuracy


def evaluate_predictions(y_true, labels, n_classes):
    acc, best_perm = clustering_accuracy(y_true, labels, n_classes)
    mapped = np.array([best_perm[label] for label in labels])
    cm = confusion_matrix(y_true, mapped)

    return {
        "accuracy": float(acc),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "best_perm": [int(x) for x in best_perm],
        "confusion_matrix": cm.tolist(),
    }


def summarize_batch_results(results):
    acc_list = [r["accuracy"] for r in results]
    ari_list = [r["ari"] for r in results]
    nmi_list = [r["nmi"] for r in results]
    ll_list = [r["final_log_likelihood"] for r in results]

    def _std(vals):
        return float(stdev(vals)) if len(vals) > 1 else 0.0

    return {
        "runs": len(results),
        "accuracy_mean": float(mean(acc_list)),
        "accuracy_std": _std(acc_list),
        "ari_mean": float(mean(ari_list)),
        "ari_std": _std(ari_list),
        "nmi_mean": float(mean(nmi_list)),
        "nmi_std": _std(nmi_list),
        "final_log_likelihood_mean": float(mean(ll_list)),
        "final_log_likelihood_std": _std(ll_list),
    }


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_batch_csv(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "seed",
        "iterations",
        "final_log_likelihood",
        "accuracy",
        "ari",
        "nmi",
        "bic",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})
