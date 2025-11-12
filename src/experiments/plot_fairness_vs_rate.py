from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = "results/fixed_rate_comparison.csv"
OUTPUT_DP = "results/fairness_vs_rate_dp.png"
OUTPUT_EO = "results/fairness_vs_rate_eo.png"


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing fixed-rate CSV: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df_sorted = df.sort_values("target_rate")

    models = ["GLM", "NN", "ADV_NN"]
    colors = ["tab:orange", "tab:green", "tab:purple"]

    # DP ratio plot
    plt.figure(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted["model_name"] == model]
        plt.plot(
            subset["target_rate"],
            subset["dp_ratio"],
            marker="o",
            color=color,
            label=model,
        )
    plt.xlabel("Approval Rate")
    plt.ylabel("DP Ratio")
    plt.title("DP Ratio vs Approval Rate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(OUTPUT_DP) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DP, dpi=200)
    plt.close()

    # EO gap plot
    plt.figure(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted["model_name"] == model]
        plt.plot(
            subset["target_rate"],
            subset["eo_gap_tpr"],
            marker="o",
            color=color,
            label=model,
        )
    plt.xlabel("Approval Rate")
    plt.ylabel("EO TPR Difference")
    plt.title("Equalized Odds Gap vs Approval Rate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_EO, dpi=200)
    plt.close()
    print(f"Saved plots to {OUTPUT_DP} and {OUTPUT_EO}")


if __name__ == "__main__":
    main()
