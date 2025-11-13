from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DP = Path("results/fairness_vs_rate_dp.png")
OUTPUT_EO = Path("results/fairness_vs_rate_eo.png")


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
    raise FileNotFoundError(f"Missing metrics for {experiment}")


def main() -> None:
    df = pd.read_csv(_find_metrics("fixed_rate_comparison"))
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
    OUTPUT_DP.parent.mkdir(parents=True, exist_ok=True)
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
