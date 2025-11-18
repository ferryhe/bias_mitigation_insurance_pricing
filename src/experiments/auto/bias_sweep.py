from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from src.auto import generate_auto_insurance_data
from src.config import get_default_auto_simulation_config, get_default_configs
from src.credit import train_test_split_df
from src.experiments.auto_baseline_utils import AUTO_FEATURE_SPEC, run_auto_models


BIAS_STRENGTHS = [0.0, 0.25, 0.5, 1.0, 2.0]


def _build_bias_table(df: pd.DataFrame, bias: float) -> list[str]:
    lines = [
        "| Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    subset = df[df["bias_strength"] == bias]
    for model in ("GLM", "NN", "ADV_NN"):
        row = subset[subset["model_name"] == model]
        if row.empty:
            continue
        series = row.iloc[0]
        lines.append(
            f"| {model} | {series['roc_auc']:.3f} | {series['eo_gap_tpr']:.3f} | {series['eo_gap_fpr']:.3f} | {series['dp_ratio']:.2f} |"
        )
    return lines


def main() -> None:
    base_sim_cfg = get_default_auto_simulation_config()
    _, train_cfg, eval_cfg = get_default_configs()
    eval_cfg.target_acceptance_rate = 0.02

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("results") / "auto" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []

    for bias_strength in BIAS_STRENGTHS:
        sim_cfg = replace(base_sim_cfg, bias_strength=bias_strength)
        df = generate_auto_insurance_data(
            sim_cfg,
        )
        df_train, df_test = train_test_split_df(
            df, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = run_auto_models(
            df_train,
            df_test,
            sim_cfg,
            train_cfg,
            eval_cfg,
            device,
            feature_spec=AUTO_FEATURE_SPEC,
        )
        for metrics in results:
            rows.append(
                {
                    "bias_strength": bias_strength,
                    "model_name": metrics["model_name"],
                    "roc_auc": metrics.get("roc_auc", np.nan),
                    "eo_gap_tpr": metrics.get("eo_gap_tpr", np.nan),
                    "eo_gap_fpr": metrics.get("eo_gap_fpr", np.nan),
                    "dp_ratio_fixed_2pct": metrics.get("dp_ratio_fixed_r", np.nan),
                    "dp_diff": metrics.get("dp_diff", np.nan),
                    "dp_ratio": metrics.get("dp_ratio", np.nan),
                }
            )

    df = pd.DataFrame(rows)
    output_path = run_root / "bias_sweep_metrics.csv"
    df.to_csv(output_path, index=False)

    readme_path = run_root / "README.md"
    def _row(df: pd.DataFrame, bias: float, model: str) -> pd.Series:
        subset = df[(df["bias_strength"] == bias) & (df["model_name"] == model)]
        if subset.empty:
            raise ValueError(f"Missing data for {model} at bias {bias}")
        return subset.iloc[0]

    def _trend_text(name: str, low: float, high: float) -> str:
        if abs(high - low) < 1e-4:
            return f"{name} stays near {low:.3f}"
        verb = "rises" if high > low else "falls"
        return f"{name} {verb} from {low:.3f} to {high:.3f}"

    readme_lines = [
        f"# Run {timestamp} - Auto bias sweep",
        "",
        "**Purpose**",
        "- Measure how the artificial `bias_strength` parameter propagates into accuracy/fairness gaps for each baseline model.",
        "- Store the sweep summary next to the credit-style baseline runs so downstream scripts can point at a canonical auto run.",
        "",
        "**Configuration**",
        f"- Bias strengths: {BIAS_STRENGTHS}",
        "- `AutoSimulationConfig` defaults (60k rows, seed 202, target claim rate 10%) with `bias_strength` injected in `claim_indicator` generation.",
        "- Training uses `TrainingConfig` defaults (batch 1024, 10 epochs with 5 warm-up epochs, `lambda_adv = 0.1`).",
        "- Evaluation inspects both threshold 0.5 plus an implicit acceptance-rate slice recorded in `bias_sweep_metrics.csv`.",
        "",
        "## Metrics by bias scenario",
        "",
        "### Bias strength = 0.0",
        "",
        * _build_bias_table(df, BIAS_STRENGTHS[0]),
        "",
        "### Bias strength = 2.0",
        "",
        * _build_bias_table(df, BIAS_STRENGTHS[-1]),
    ]

    observation_lines = ["", "**Observations**"]
    low_bias = BIAS_STRENGTHS[0]
    high_bias = BIAS_STRENGTHS[-1]
    for model in ("GLM", "NN", "ADV_NN"):
        low_row = _row(df, low_bias, model)
        high_row = _row(df, high_bias, model)
        observation_lines.append(
            f"- {model}: {_trend_text('EO ΔTPR', low_row['eo_gap_tpr'], high_row['eo_gap_tpr'])}; "
            f"{_trend_text('DP ratio', low_row['dp_ratio'], high_row['dp_ratio'])}."
        )
    observation_lines.append(
        "- The sweep log in `bias_sweep_metrics.csv` can drive plots of EO gap, DP ratio, and ROC AUC vs. injected bias strength."
    )
    observation_lines.append(
        "- Repeat the sweep after tuning `lambda_adv` or acceptance rates to compare how mitigation scales with stronger generative bias."
    )
    readme_lines.extend(observation_lines)
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Saved bias sweep metrics and README under {run_root}")


if __name__ == "__main__":
    main()
