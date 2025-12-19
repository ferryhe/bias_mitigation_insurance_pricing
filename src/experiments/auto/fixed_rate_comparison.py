from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from src.auto import generate_auto_insurance_data
from src.config import get_default_auto_simulation_config, get_default_configs
from src.credit import train_test_split_df
from src.experiments.auto_baseline_utils import AUTO_FEATURE_SPEC, run_auto_models

TARGET_RATES = [0.01, 0.02, 0.05]
ADV_LAMBDA = 2.0


def main() -> None:
    sim_cfg = get_default_auto_simulation_config()
    _, train_cfg, eval_cfg = get_default_configs()

    df = generate_auto_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(
        df, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    for target_rate in TARGET_RATES:
        eval_case = replace(eval_cfg, target_acceptance_rate=target_rate)
        train_case = replace(train_cfg, lambda_adv=ADV_LAMBDA)
        results = run_auto_models(
            df_train,
            df_test,
            sim_cfg,
            train_case,
            eval_case,
            device,
            feature_spec=AUTO_FEATURE_SPEC,
        )
        for metrics in results:
            record = {"target_rate": target_rate, **metrics}
            if record["model_name"] == "ADV_NN":
                record["lambda_adv"] = ADV_LAMBDA
            rows.append(record)

    df_results = pd.DataFrame(rows)
    out_path = Path("results/auto/fixed_rate_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"Saved fixed-rate comparison to {out_path}")


if __name__ == "__main__":
    main()
