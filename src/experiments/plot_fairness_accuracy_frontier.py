from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


LAMBDA_SWEEP_PATH = "results/lambda_sweep.csv"
FIXED_RATE_PATH = "results/fixed_rate_comparison.csv"
OUTPUT_PATH = "results/fairness_accuracy_frontier.png"


def main() -> None:
    if not os.path.exists(LAMBDA_SWEEP_PATH):
        raise FileNotFoundError(f"Missing lambda sweep CSV: {LAMBDA_SWEEP_PATH}")
    if not os.path.exists(FIXED_RATE_PATH):
        raise FileNotFoundError(f"Missing fixed-rate CSV: {FIXED_RATE_PATH}")

    df_lambda = pd.read_csv(LAMBDA_SWEEP_PATH)
    df_fixed = pd.read_csv(FIXED_RATE_PATH)

    plt.figure(figsize=(7, 5))
    plt.grid(True, linestyle="--", alpha=0.5)

    # Lambda sweep points
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

    # Baseline GLM / NN
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

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
