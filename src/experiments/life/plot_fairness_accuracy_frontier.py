from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_PATH = Path("results/life/fairness_accuracy_frontier.png")


def _find_metrics(experiment: str) -> Path:
    root = Path("results")
    candidates = [
        root / "life" / f"{experiment}.csv",
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
        candidates.append(run_dir / "life" / target_folder / "metrics.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find metrics for {experiment}")


def _pick(df: pd.DataFrame, names: Iterable[str], default: str | None = None) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return default


def _standardize_fixed(df_fixed: pd.DataFrame) -> pd.DataFrame:
    df = df_fixed.copy()
    if "model_name" not in df.columns and "model" in df.columns:
        df["model_name"] = df["model"]
    if "eo_gap_tpr" not in df.columns:
        for alt in ("eo_gap_tpr_fixed_r", "eo_gap_tpr_fixed"):
            if alt in df.columns:
                df["eo_gap_tpr"] = df[alt]
                break
    return df


def _standardize_lambda(df_lambda: pd.DataFrame) -> pd.DataFrame:
    df = df_lambda.copy()
    if "eo_gap_tpr" not in df.columns:
        for alt in ("eo_gap_tpr_fixed_r", "eo_gap_tpr_fixed"):
            if alt in df.columns:
                df["eo_gap_tpr"] = df[alt]
                break
    return df


def plot_fairness_accuracy_frontier(
    df_lambda: pd.DataFrame,
    df_fixed: pd.DataFrame,
    title: str = "Fairness vs Accuracy (Equalized Odds)",
) -> Path:
    df_lambda = _standardize_lambda(df_lambda) if not df_lambda.empty else df_lambda
    df_fixed = _standardize_fixed(df_fixed)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True, linestyle="--", alpha=0.5)

    if not df_lambda.empty:
        eo_col = _pick(df_lambda, ("eo_gap_tpr",))
        auc_col = _pick(df_lambda, ("roc_auc",))
        if eo_col and auc_col:
            lambda_col = _pick(df_lambda, ("lambda_adv", "lambda"))
            ax.scatter(df_lambda[eo_col], df_lambda[auc_col], c="tab:blue", label="Adv NN lambda sweep")
            for idx, row in df_lambda.iterrows():
                label_val = row[lambda_col] if lambda_col and lambda_col in row else f"idx={idx}"
                ax.annotate(
                    f"lambda={label_val}",
                    (row[eo_col], row[auc_col]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    eo_col_f = _pick(df_fixed, ("eo_gap_tpr",))
    auc_col_f = _pick(df_fixed, ("roc_auc",))
    model_col = _pick(df_fixed, ("model_name", "model"))
    if eo_col_f is None or auc_col_f is None or model_col is None:
        raise KeyError("df_fixed must contain eo_gap_tpr/roc_auc and model_name columns")
    for model_name, color in [("GLM", "tab:orange"), ("NN", "tab:green"), ("ADV_NN", "tab:purple")]:
        subset = df_fixed[df_fixed[model_col] == model_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        ax.scatter(row[eo_col_f], row[auc_col_f], c=color, marker="D", label=model_name)
        ax.annotate(
            model_name,
            (row[eo_col_f], row[auc_col_f]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel("EO TPR difference")
    ax.set_ylabel("ROC AUC")
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return OUTPUT_PATH


def main(
    df_lambda: pd.DataFrame | None = None,
    df_fixed: pd.DataFrame | None = None,
) -> Path:
    if df_lambda is None:
        try:
            df_lambda = pd.read_csv(_find_metrics("lambda_sweep"))
        except FileNotFoundError:
            df_lambda = pd.DataFrame()
    if df_fixed is None:
        df_fixed = pd.read_csv(_find_metrics("fixed_rate_comparison"))

    path = plot_fairness_accuracy_frontier(df_lambda, df_fixed)
    print(f"Saved plot to {path}")
    return path


if __name__ == "__main__":
    main()
