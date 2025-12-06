from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_PATH = Path("results/auto/fairness_accuracy_frontier.png")


def _find_metrics(filename: str) -> Path:
    root = Path("results/auto")
    direct = root / filename
    if direct.exists():
        return direct

    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under results/auto")


def plot_fairness_accuracy_frontier(df_bias: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True, linestyle="--", alpha=0.5)

    colors = {"GLM": "tab:orange", "NN": "tab:green", "ADV_NN": "tab:purple"}
    for model_name, color in colors.items():
        subset = df_bias[df_bias["model_name"] == model_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["eo_gap_tpr"],
            subset["roc_auc"],
            label=model_name,
            color=color,
            alpha=0.85,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"bias={row['bias_strength']}",
                (row["eo_gap_tpr"], row["roc_auc"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_title("Auto fairness vs accuracy (Equalized Odds)")
    ax.set_xlabel("EO TPR difference")
    ax.set_ylabel("ROC AUC")
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return OUTPUT_PATH


def main() -> Path:
    bias_df = pd.read_csv(_find_metrics("bias_sweep_metrics.csv"))
    path = plot_fairness_accuracy_frontier(bias_df)
    print(f"Saved plot to {path}")
    return path


if __name__ == "__main__":
    main()
