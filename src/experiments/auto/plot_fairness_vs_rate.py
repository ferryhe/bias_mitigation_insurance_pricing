from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DP = Path("results/auto/fairness_vs_bias_dp.png")
OUTPUT_EO = Path("results/auto/fairness_vs_bias_eo.png")


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
    raise FileNotFoundError(f"Missing {filename} under results/auto")


def plot_fairness_vs_rate(df_bias: pd.DataFrame) -> tuple[Path, Path]:
    df_sorted = df_bias.sort_values("bias_strength")
    models = ["GLM", "NN", "ADV_NN"]
    colors = ["tab:orange", "tab:green", "tab:purple"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, colors):
        subset = df_sorted[df_sorted["model_name"] == model]
        if subset.empty:
            continue
        ax.plot(
            subset["bias_strength"],
            subset["dp_ratio"],
            marker="o",
            color=color,
            label=model,
        )
    ax.set_xlabel("Bias strength")
    ax.set_ylabel("DP ratio")
    ax.set_title("DP ratio vs injected bias strength")
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
            subset["bias_strength"],
            subset["eo_gap_tpr"],
            marker="o",
            color=color,
            label=model,
        )
    ax.set_xlabel("Bias strength")
    ax.set_ylabel("EO TPR difference")
    ax.set_title("Equalized odds gap vs injected bias strength")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_EO, dpi=200)
    plt.close(fig)

    return OUTPUT_DP, OUTPUT_EO


def main() -> tuple[Path, Path]:
    bias_df = pd.read_csv(_find_metrics("bias_sweep_metrics.csv"))
    dp_path, eo_path = plot_fairness_vs_rate(bias_df)
    print(f"Saved plots to {dp_path} and {eo_path}")
    return dp_path, eo_path


if __name__ == "__main__":
    main()
