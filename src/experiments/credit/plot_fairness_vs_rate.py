from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DP = Path("results/credit/fairness_vs_rate_dp.png")
OUTPUT_EO = Path("results/credit/fairness_vs_rate_eo.png")


def _find_metrics(experiment: str) -> Path:
    root = Path("results")
    candidates = [
        root / "credit" / f"{experiment}.csv",
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
        candidates.append(run_dir / "credit" / target_folder / "metrics.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing metrics for {experiment}")


def plot_fairness_vs_rate(df_fixed: pd.DataFrame) -> tuple[Path, Path]:
    df_sorted = df_fixed.sort_values("target_rate")
    models = ["GLM", "NN", "ADV_NN"]
    colors = ["tab:orange", "tab:green", "tab:purple"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted["model_name"] == model]
        if subset.empty:
            continue
        ax.plot(
            subset["target_rate"],
            subset["dp_ratio"],
            marker="o",
            color=color,
            label=model,
        )
    ax.set_xlabel("High-risk rate")
    ax.set_ylabel("DP ratio")
    ax.set_title("DP ratio vs high-risk rate")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    OUTPUT_DP.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DP, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted["model_name"] == model]
        if subset.empty:
            continue
        ax.plot(
            subset["target_rate"],
            subset["eo_gap_tpr"],
            marker="o",
            color=color,
            label=model,
        )
    ax.set_xlabel("High-risk rate")
    ax.set_ylabel("EO TPR difference")
    ax.set_title("Equalized odds gap vs high-risk rate")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_EO, dpi=200)
    plt.close(fig)

    return OUTPUT_DP, OUTPUT_EO


def main() -> tuple[Path, Path]:
    df_fixed = pd.read_csv(_find_metrics("fixed_rate_comparison"))
    dp_path, eo_path = plot_fairness_vs_rate(df_fixed)
    print(f"Saved plots to {dp_path} and {eo_path}")
    return dp_path, eo_path


if __name__ == "__main__":
    main()
