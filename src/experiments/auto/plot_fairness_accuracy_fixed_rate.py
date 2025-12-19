from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_PATH = Path("results/auto/fairness_accuracy_fixed_rate.png")


def _find_metrics() -> Path:
    root = Path("results")
    candidates = [
        root / "auto" / "fixed_rate_comparison.csv",
        root / "fixed_rate_comparison.csv",
    ]
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        candidates.append(run_dir / "auto" / "fixed_rate_comparison" / "metrics.csv")
        candidates.append(run_dir / "fixed_rate_comparison" / "metrics.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find fixed_rate_comparison metrics")


def plot_frontier(df_fixed: pd.DataFrame) -> Path:
    df_2pct = df_fixed[df_fixed["target_rate"] == 0.02]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True, linestyle="--", alpha=0.5)

    colors = {"GLM": "tab:orange", "NN": "tab:green", "ADV_NN": "tab:purple"}
    for model_name, color in colors.items():
        subset = df_2pct[df_2pct["model_name"] == model_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        ax.scatter(row["eo_gap_tpr_fixed_r"], row["roc_auc"], c=color, label=model_name, marker="D")
        ax.annotate(
            model_name,
            (row["eo_gap_tpr_fixed_r"], row["roc_auc"]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=9,
            fontweight="bold",
            color=color,
        )
    ax.set_title("Fairness vs Accuracy (2% high-risk slice)")
    ax.set_xlabel("EO TPR difference @2%")
    ax.set_ylabel("ROC AUC")
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return OUTPUT_PATH


def main() -> Path:
    df_fixed = pd.read_csv(_find_metrics())
    path = plot_frontier(df_fixed)
    print(f"Saved plot to {path}")
    return path


if __name__ == "__main__":
    main()
