from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DP = Path("results/life/fairness_vs_rate_dp.png")
OUTPUT_EO = Path("results/life/fairness_vs_rate_eo.png")


def _find_metrics(experiment: str) -> Path:
    root = Path("results")
    candidates = [
        root / "life" / f"{experiment}.csv",
        root / f"{experiment}.csv",
    ]
    folder_alias = {
        "fixed_rate_comparison": "fixed_rate",
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
    raise FileNotFoundError(f"Missing metrics for {experiment}")


def _pick(df: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Expected one of {names} in DataFrame columns")


def plot_fairness_vs_rate(df_fixed: pd.DataFrame) -> tuple[Path, Path]:
    df_sorted = df_fixed.sort_values(_pick(df_fixed, ("target_rate",)))
    model_col = _pick(df_sorted, ("model_name", "model"))
    dp_col = _pick(df_sorted, ("dp_ratio_fixed_r", "dp_ratio"))
    eo_col = _pick(df_sorted, ("eo_gap_tpr_fixed_r", "eo_gap_tpr"))
    target_col = _pick(df_sorted, ("target_rate",))

    models = ["GLM", "NN", "ADV_NN"]
    colors = ["tab:orange", "tab:green", "tab:purple"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted[model_col] == model]
        if subset.empty:
            continue
        ax.plot(subset[target_col], subset[dp_col], marker="o", color=color, label=model)
    ax.set_xlabel("High-risk rate")
    ax.set_ylabel("DP ratio (at target rate)")
    ax.set_title("DP ratio vs high-risk rate")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    OUTPUT_DP.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DP, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted[model_col] == model]
        if subset.empty:
            continue
        ax.plot(subset[target_col], subset[eo_col], marker="o", color=color, label=model)
    ax.set_xlabel("High-risk rate")
    ax.set_ylabel("EO TPR difference (at target rate)")
    ax.set_title("Equalized odds gap vs high-risk rate")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_EO, dpi=200)
    plt.close(fig)

    return OUTPUT_DP, OUTPUT_EO


def main(df_fixed: pd.DataFrame | None = None) -> tuple[Path, Path]:
    if df_fixed is None:
        df_fixed = pd.read_csv(_find_metrics("fixed_rate_comparison"))
    return plot_fairness_vs_rate(df_fixed)


if __name__ == "__main__":
    main()
