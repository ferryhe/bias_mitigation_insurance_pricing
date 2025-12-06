from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _sigma(x: np.ndarray) -> np.ndarray:
    """Logistic link σ(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_health_underwriting_data(
    n_samples: int = 100_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate a health insurance underwriting dataset per Health DGP (Section B).

    Returns a DataFrame with columns:
        ['A18_34','A35_49','A50_64','female','chronic',
         'prior_cost','ADI_high','Y_post',
         'Race','T_true','H0'].
    """

    rng = np.random.default_rng(seed)
    sigma = 0.35

    # 3.2 Race A/B and ADI_high
    race_is_a = rng.binomial(1, 0.25, size=n_samples)
    p_adi = np.where(race_is_a == 1, 0.75, 0.25)
    adi_high = rng.binomial(1, p_adi)

    # 3.3 Age bands, female, chronic
    age_multinomial = rng.multinomial(1, [0.40, 0.35, 0.25], size=n_samples)
    a18_34 = age_multinomial[:, 0]
    a35_49 = age_multinomial[:, 1]
    a50_64 = age_multinomial[:, 2]

    female = rng.binomial(1, 0.50, size=n_samples)

    p_chronic = np.where(race_is_a == 1, 0.35, 0.30)
    chronic = rng.binomial(1, p_chronic)

    # --- Non-linear latent risk tier (NOT in the feature set) ---
    # Base linear score combining age, chronic, and ADI_high
    risk_score = (
        0.4 * a35_49
        + 0.9 * a50_64
        + 0.8 * chronic
        + 0.7 * adi_high
        + 0.8 * adi_high * chronic      # strong interaction
        + 0.6 * adi_high * a50_64       # strong interaction
    )

    # Add some noise so tiers are not perfectly deterministic
    risk_score = risk_score + rng.normal(0.0, 0.4, size=n_samples)

    # Threshold into 3 tiers: 0 (low), 1 (medium), 2 (high)
    risk_tier = np.zeros(n_samples, dtype=int)
    risk_tier[risk_score > 1.0] = 1
    risk_tier[risk_score > 2.0] = 2


    # 3.4 True need T_true
    eta = rng.normal(0.0, sigma, size=n_samples)

    # Strongly non-linear in the (unobserved) risk_tier
    log_t = (
        np.log(3000.0)
        # modest main effects
        + 0.15 * a35_49
        + 0.40 * a50_64
        + 0.05 * female
        + 0.50 * chronic
        + 0.20 * adi_high
        + 0.05 * race_is_a
        # non-linear tier effects
        + 0.50 * risk_tier              # linear tier effect
        + 0.40 * (risk_tier == 2)       # extra kicker for top tier
        + eta
    )

    t_true = np.exp(log_t)
        

    # 3.5 Pre-policy access H0
    xi = rng.normal(0.0, sigma, size=n_samples)
    s0 = -1.5 + 1.2 * adi_high + 0.30 * race_is_a + xi
    u = _sigma(-s0)
    h0 = np.clip(_sigma(-s0), 0.15, 1.0)

    # 3.6 Observed costs
    eps0 = np.exp(rng.normal(0.0, sigma, size=n_samples))
    eps1 = np.exp(rng.normal(0.0, sigma, size=n_samples))

    y_pre = h0 * t_true * eps0
    prior_cost = y_pre
    y_post = t_true * eps1

    race_labels = np.where(race_is_a == 1, "A", "B")

    return pd.DataFrame(
        {
            "A18_34": a18_34,
            "A35_49": a35_49,
            "A50_64": a50_64,
            "female": female,
            "chronic": chronic,
            "prior_cost": prior_cost,
            "ADI_high": adi_high,
            "Y_post": y_post,
            "Race": race_labels,
            "T_true": t_true,
            "H0": h0,
        }
    )


def train_test_split_health(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple random split (default 80/20) without external deps.

    Returns (df_train, df_test).
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(df)
    all_idx = np.arange(n)
    shuffled = rng.permutation(all_idx)
    n_test = int(np.round(test_size * n))

    test_idx = shuffled[:n_test]
    train_idx = shuffled[n_test:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test
