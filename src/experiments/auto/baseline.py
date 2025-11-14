from __future__ import annotations

import os
from datetime import datetime

import torch

from pathlib import Path

from src.auto import generate_auto_insurance_data
from src.common.feature_spec import FeatureSpec
from src.config import get_default_configs, get_default_auto_simulation_config
from src.credit import train_test_split_df
from src.evaluation.reporting import format_metrics_table, save_metrics
from src.experiments.auto_baseline_utils import AUTO_FEATURE_SPEC, run_auto_models


AUTO_FEATURE_SPEC = FeatureSpec(
    numeric_features=(
        "territory",
        "age",
        "years_licensed",
        "annual_mileage",
        "vehicle_use",
        "vehicle_age",
        "vehicle_value",
        "safety_score",
        "past_claims_obs",
        "violations_obs",
        "credit_score",
        "income",
    ),
    protected_feature="A",
    proxy_feature="territory",
    target_feature="claim_indicator",
)


def main() -> None:
    sim_cfg = get_default_auto_simulation_config()
    _, train_cfg, eval_cfg = get_default_configs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path("results") / "auto" / timestamp
    debug_dir = results_root / "debug_auto_distributions"
    df = generate_auto_insurance_data(
        sim_cfg,
        debug_plots=True,
        debug_plots_dir=debug_dir,
    )
    df_train, df_test = train_test_split_df(
        df, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg.target_acceptance_rate = 0.02

    results = run_auto_models(
        df_train,
        df_test,
        sim_cfg,
        train_cfg,
        eval_cfg,
        device,
        feature_spec=AUTO_FEATURE_SPEC,
    )

    metrics_table = format_metrics_table(results)
    print(metrics_table)

    os.makedirs(results_root, exist_ok=True)
    save_path = results_root / "baseline_results.csv"
    save_metrics(metrics_table, save_path)


if __name__ == "__main__":
    main()
