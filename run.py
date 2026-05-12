import numpy as np
import os
from pathlib import Path

from config import BatchConfig, ExperimentConfig
from data_utils import class_distribution, load_and_preprocess_iris
from em_gmm import GaussianMixtureEM
from evaluation import (
    evaluate_predictions,
    save_batch_csv,
    save_json,
    summarize_batch_results,
)


REQUIRED_FILES = [
    "em_gmm.py",
    "config.py",
    "data_utils.py",
    "evaluation.py",
    "run.py",
    "README.md",
    "requirements.txt",
    "outputs/single_run_report.json",
    "outputs/batch_summary.json",
    "outputs/batch_results.csv",
    "outputs/model_selection.json",
]

BASE_DIR = Path(__file__).resolve().parent


def check_files():
    missing = [p for p in REQUIRED_FILES if not (BASE_DIR / p).exists()]
    if missing:
        print("Missing files:")
        for p in missing:
            print(f"  - {p}")
        return False
    return True


def run_single(x, y, feature_names, target_names, cfg):
    n_classes = len(np.unique(y))
    model = GaussianMixtureEM(
        n_components=cfg.n_components,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        random_state=cfg.random_state,
    )
    result = model.fit_predict(x)
    metrics = evaluate_predictions(y, result.labels, n_classes)
    final_bic = model.bic(x)

    report = {
        "dataset": cfg.dataset_name,
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "feature_names": feature_names,
        "target_names": target_names.tolist(),
        "class_distribution": class_distribution(y),
        "iterations": len(result.log_likelihood_history),
        "final_log_likelihood": float(result.log_likelihood_history[-1]),
        "bic": float(final_bic),
        "metrics": metrics,
        "weights": np.round(result.weights, 6).tolist(),
        "means": np.round(result.means, 6).tolist(),
    }
    save_json("outputs/single_run_report.json", report)

    print("=== GMM-EM Experiment on Iris ===")
    print(f"Samples: {x.shape[0]}, Features: {x.shape[1]}")
    print(f"Components: {n_classes}")
    print(f"EM iterations: {len(result.log_likelihood_history)}")
    print(f"Final log-likelihood: {result.log_likelihood_history[-1]:.6f}")
    print(f"BIC: {final_bic:.4f}")
    print(f"Best label permutation: {tuple(metrics['best_perm'])}")
    print(f"Clustering accuracy: {metrics['accuracy']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")


def run_batch(x, y, exp_cfg, batch_cfg):
    n_classes = len(np.unique(y))
    seeds = range(batch_cfg.seed_start, batch_cfg.seed_start + batch_cfg.n_runs)
    results = []
    for seed in seeds:
        model = GaussianMixtureEM(
            n_components=exp_cfg.n_components,
            max_iter=exp_cfg.max_iter,
            tol=exp_cfg.tol,
            random_state=seed,
        )
        result = model.fit_predict(x)
        metrics = evaluate_predictions(y, result.labels, n_classes)
        results.append(
            {
                "seed": int(seed),
                "iterations": int(len(result.log_likelihood_history)),
                "final_log_likelihood": float(result.log_likelihood_history[-1]),
                "accuracy": float(metrics["accuracy"]),
                "ari": float(metrics["ari"]),
                "nmi": float(metrics["nmi"]),
                "bic": float(model.bic(x)),
            }
        )

    summary = summarize_batch_results(results)
    output = {
        "dataset": exp_cfg.dataset_name,
        "n_components": exp_cfg.n_components,
        "max_iter": exp_cfg.max_iter,
        "tol": exp_cfg.tol,
        "seed_start": batch_cfg.seed_start,
        "n_runs": batch_cfg.n_runs,
        "summary": summary,
        "runs": results,
    }
    save_json("outputs/batch_summary.json", output)
    save_batch_csv("outputs/batch_results.csv", results)

    print("=== Batch GMM-EM Experiments ===")
    print(f"Runs: {summary['runs']}")
    print(
        "Accuracy mean/std: "
        f"{summary['accuracy_mean']:.4f} / {summary['accuracy_std']:.4f}"
    )
    print(f"ARI mean/std: {summary['ari_mean']:.4f} / {summary['ari_std']:.4f}")
    print(f"NMI mean/std: {summary['nmi_mean']:.4f} / {summary['nmi_std']:.4f}")
    print(
        "Final log-likelihood mean/std: "
        f"{summary['final_log_likelihood_mean']:.4f} / "
        f"{summary['final_log_likelihood_std']:.4f}"
    )


def run_model_selection(x, cfg):
    candidates = [2, 3, 4, 5]
    rows = []
    for k in candidates:
        model = GaussianMixtureEM(
            n_components=k,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            random_state=cfg.random_state,
        )
        result = model.fit_predict(x)
        rows.append(
            {
                "k": int(k),
                "iterations": int(len(result.log_likelihood_history)),
                "final_log_likelihood": float(result.log_likelihood_history[-1]),
                "bic": float(model.bic(x)),
            }
        )
    best = min(rows, key=lambda r: r["bic"])
    payload = {"dataset": cfg.dataset_name, "candidates": rows, "best_by_bic": best}
    save_json("outputs/model_selection.json", payload)

    print("=== Model Selection by BIC ===")
    for row in rows:
        print(
            f"K={row['k']}: BIC={row['bic']:.4f}, "
            f"LL={row['final_log_likelihood']:.4f}, Iter={row['iterations']}"
        )
    print(f"Best K by BIC: {best['k']}")


def main():
    # Make all relative paths stable no matter where python is launched.
    os.chdir(BASE_DIR)

    exp_cfg = ExperimentConfig()
    batch_cfg = BatchConfig()
    x, y, feature_names, target_names = load_and_preprocess_iris()

    run_single(x, y, feature_names, target_names, exp_cfg)
    run_batch(x, y, exp_cfg, batch_cfg)
    run_model_selection(x, exp_cfg)

    ok = check_files()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
