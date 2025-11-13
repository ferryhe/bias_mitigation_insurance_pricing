from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_PATH = Path("results/fairness_accuracy_frontier.png")


def _find_metrics(experiment: str) -> Path:
    root = Path("results")
    direct = root / f"{experiment}.csv"
    if direct.exists():
        return direct
    folder_alias = {
        "fixed_rate_comparison": "fixed_rate",
        "lambda_sweep": "lambda_sweep",
    }
    target_folder = folder_alias.get(experiment, experiment)
    for run_dir in sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    ):
        candidate = run_dir / target_folder / "metrics.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find metrics for {experiment}")


def main() -> None:
    df_lambda = pd.read_csv(_find_metrics("lambda_sweep"))
    df_fixed = pd.read_csv(_find_metrics("fixed_rate_comparison"))

    plt.figure(figsize=(7, 5))
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.scatter(
        df_lambda["eo_gap_tpr"],
        df_lambda["roc_auc"],
        c="tab:blue",
        label="Adversarial λ sweep",
    )
    for _, row in df_lambda.iterrows():
        plt.annotate(
            f"λ={row['lambda_adv']}",
            (row["eo_gap_tpr"], row["roc_auc"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    for model_name, color in [("GLM", "tab:orange"), ("NN", "tab:green")]:
        row = df_fixed[df_fixed["model_name"] == model_name].iloc[0]
        plt.scatter(
            row["eo_gap_tpr"],
            row["roc_auc"],
            c=color,
            marker="D",
            label=model_name,
        )
        plt.annotate(
            model_name,
            (row["eo_gap_tpr"], row["roc_auc"]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    plt.title("Fairness–Accuracy Frontier (Equalized Odds)")
    plt.xlabel("EO TPR Difference")
    plt.ylabel("ROC AUC")
    plt.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
