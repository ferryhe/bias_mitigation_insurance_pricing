from __future__ import annotations

import pandas as pd
import torch
from dataclasses import replace
from pathlib import Path

from src.auto import generate_auto_insurance_data
from src.config import get_default_auto_simulation_config, get_default_configs
from src.credit import train_test_split_df
from src.experiments.auto_baseline_utils import AUTO_FEATURE_SPEC
from src.training.train_adv_nn import train_and_eval_adv_nn

LAMBDAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]


def main() -> None:
    sim_cfg = get_default_auto_simulation_config()
    train_cfg, eval_cfg = get_default_configs()[1:]

    df = generate_auto_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(
        df, test_size=0.2, seed=sim_cfg.seed, target_col="claim_indicator"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg.target_acceptance_rate = 0.02

    rows: list[dict[str, float]] = []
    for lambda_val in LAMBDAS:
        cfg = replace(train_cfg, lambda_adv=lambda_val)
        metrics = train_and_eval_adv_nn(
            df_train,
            df_test,
            sim_cfg,
            cfg,
            eval_cfg,
            device,
            feature_spec=AUTO_FEATURE_SPEC,
        )
        metrics = metrics.copy()
        metrics["lambda_adv"] = lambda_val
        rows.append(metrics)

    sweep_df = pd.DataFrame(rows)
    out_path = Path("results/auto/lambda_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_path, index=False)
    print(f"Saved lambda sweep to {out_path}")


if __name__ == "__main__":
    main()
