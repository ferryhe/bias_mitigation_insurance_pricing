from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _sigma(x: np.ndarray) -> np.ndarray:
    """Logistic link σ(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    size: int,
) -> np.ndarray:
    """Rejection-sample N(mean, std^2) truncated to [lower, upper] (A.3.2)."""
    samples = rng.normal(loc=mean, scale=std, size=size)
    mask = (samples < lower) | (samples > upper)
    while mask.any():
        resamples = rng.normal(loc=mean, scale=std, size=mask.sum())
        samples[mask] = resamples
        mask = (samples < lower) | (samples > upper)
    return samples


def _calibrate_intercept(
    eta_no_intercept: np.ndarray,
    target_rate: float = 0.15,
    lower: float = -6.0,
    upper: float = 6.0,
    n_iter: int = 50,
) -> float:
    """Bisection solve for β_0 so mean σ(β_0 + η) ≈ target_rate (A.3.6)."""
    b_l, b_u = lower, upper
    for _ in range(n_iter):
        b_m = 0.5 * (b_l + b_u)
        p_bar = _sigma(b_m + eta_no_intercept).mean()
        if p_bar < target_rate:
            b_l = b_m
        else:
            b_u = b_m
    return 0.5 * (b_l + b_u)


def generate_auto_underwriting_data(
    n_samples: int = 100_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate an auto insurance underwriting dataset per Auto DGP (Section A).

    Returns columns:
    ['Age', 'M_h', 'V', 'P_prior', 'T', 'Y', 'Race', 'V_star'].
    """

    rng = np.random.default_rng(seed)

    # A.3.1 Race and Territory
    race_is_a = rng.binomial(1, 0.35, size=n_samples)
    p_t = np.where(race_is_a == 1, 0.60, 0.20)
    territory = rng.binomial(1, p_t)

    # A.3.2 Age ~ TruncNorm(40, 12; [18,85])
    age = _truncated_normal(rng, mean=40.0, std=12.0, lower=18.0, upper=85.0, size=n_samples)
    y_young = (age < 25.0).astype(int)

    # A.3.3 Mileage M ~ Gamma(k=5.76, θ≈2.0833); transform M_h = log(1 + M/12)
    mileage = rng.gamma(shape=5.76, scale=2.0833, size=n_samples)
    m_h = np.log1p(mileage / 12.0)

    # A.3.4 True violations V* with log λ* = c_0 + c_M M_h + c_Y Y_young + c_T T
    c_0, c_m, c_y, c_t = -0.65, 0.35, 0.70, 0.20
    lambda_star = np.exp(c_0 + c_m * m_h + c_y * y_young + c_t * territory)
    v_star = rng.poisson(lam=lambda_star)

    # Recorded violations V = V* + Δ, where Δ ~ Poisson(δ_T T)
    delta_t = 0.30
    delta = rng.poisson(lam=delta_t * territory)
    v_recorded = v_star + delta

    # A.3.5 Prior at-fault claim: P_prior ~ Bernoulli(σ(π_0 + π_V V* + π_T T + π_Y Y_young))
    pi_0, pi_v, pi_t, pi_y = -2.60, 0.50, 0.30, 0.40
    p_prior_prob = _sigma(pi_0 + pi_v * v_star + pi_t * territory + pi_y * y_young)
    p_prior = rng.binomial(1, p_prior_prob)

    # A.3.6 Target claim: P(Y=1) = σ(β_0 + β_M M_h + β_V V* + β_T T + β_Y Y_young + φ M_h T)
    beta_m, beta_v, beta_t, beta_y, phi = 0.55, 0.45, 0.12, 0.35, 0.15
    eta_no_intercept = (
        beta_m * m_h + beta_v * v_star + beta_t * territory + beta_y * y_young + phi * m_h * territory
    )
    beta_0 = _calibrate_intercept(eta_no_intercept, target_rate=0.15)
    p_y = _sigma(beta_0 + eta_no_intercept)
    y = rng.binomial(1, p_y)

    race_labels = np.where(race_is_a == 1, "A", "B")

    return pd.DataFrame(
        {
            "Age": age,
            "M_h": m_h,
            "V": v_recorded,
            "P_prior": p_prior,
            "T": territory,
            "Y": y,
            "Race": race_labels,
            "V_star": v_star,
        }
    )


def train_test_split_auto(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split on Y (default 80/20) without external deps.

    Returns (df_train, df_test) with the Y-class proportions preserved.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    y = df["Y"].to_numpy()
    all_idx = np.arange(len(df))

    test_indices = []
    for label in (0, 1):
        class_idx = all_idx[y == label]
        shuffled = rng.permutation(class_idx)
        n_test = int(np.round(test_size * len(class_idx)))
        test_indices.append(shuffled[:n_test])

    test_idx = np.concatenate(test_indices)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    train_idx = all_idx[train_mask]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test
