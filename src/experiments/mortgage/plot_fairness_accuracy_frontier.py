from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_PATH = Path("results/mortgage/fairness_accuracy_frontier.png")


def _find_metrics(experiment: str) -> Path:
    root = Path("results")
    candidates = [
        root / "mortgage" / f"{experiment}.csv",
        root / f"{experiment}.csv",
    ]
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
        candidates.append(run_dir / target_folder / "metrics.csv")
        candidates.append(run_dir / "mortgage" / target_folder / "metrics.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find metrics for {experiment}")


def plot_fairness_accuracy_frontier(df_lambda: pd.DataFrame, df_fixed: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True, linestyle="--", alpha=0.5)

    if not df_lambda.empty:
        ax.scatter(
            df_lambda["eo_gap_tpr"],
            df_lambda["roc_auc"],
            c="tab:blue",
            label="Adv NN lambda sweep",
        )
        for _, row in df_lambda.iterrows():
            ax.annotate(
                f"lambda={row['lambda_adv']}",
                (row["eo_gap_tpr"], row["roc_auc"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    for model_name, color in [("GLM", "tab:orange"), ("NN", "tab:green"), ("ADV_NN", "tab:purple")]:
        subset = df_fixed[df_fixed["model_name"] == model_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        ax.scatter(row["eo_gap_tpr"], row["roc_auc"], c=color, marker="D", label=model_name)
        ax.annotate(
            model_name,
            (row["eo_gap_tpr"], row["roc_auc"]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_title("Mortgage fairness vs accuracy (Equalized Odds)")
    ax.set_xlabel("EO TPR difference")
    ax.set_ylabel("ROC AUC")
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return OUTPUT_PATH


def main() -> Path:
    df_lambda = pd.read_csv(_find_metrics("lambda_sweep"))
    df_fixed = pd.read_csv(_find_metrics("fixed_rate_comparison"))
    path = plot_fairness_accuracy_frontier(df_lambda, df_fixed)
    print(f"Saved plot to {path}")
    return path


if __name__ == "__main__":
    main()
