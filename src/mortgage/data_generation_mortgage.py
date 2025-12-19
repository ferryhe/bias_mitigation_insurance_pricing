from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _sigma(x: np.ndarray) -> np.ndarray:
    """Logistic link sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    size: int,
) -> np.ndarray:
    """Rejection-sample N(mean, std^2) truncated to [lower, upper]."""
    samples = rng.normal(loc=mean, scale=std, size=size)
    mask = (samples < lower) | (samples > upper)
    while mask.any():
        resamples = rng.normal(loc=mean, scale=std, size=mask.sum())
        samples[mask] = resamples
        mask = (samples < lower) | (samples > upper)
    return samples


def _calibrate_intercept(
    eta_no_intercept: np.ndarray,
    target_rate: float = 0.10,
    lower: float = -10.0,
    upper: float = 10.0,
    n_iter: int = 50,
) -> float:
    """Bisection solve for alpha_0 so mean sigma(alpha_0 + eta) ~= target_rate."""
    b_l, b_u = lower, upper
    for _ in range(n_iter):
        b_m = 0.5 * (b_l + b_u)
        p_bar = _sigma(b_m + eta_no_intercept).mean()
        if p_bar < target_rate:
            b_l = b_m
        else:
            b_u = b_m
    return 0.5 * (b_l + b_u)


def generate_mortgage_default_data(
    n_samples: int = 100_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate the mortgage default DGP (Section D) with measurement bias in FICO.

    Returns columns in order:
    ['FICO', 'LTV', 'DTI', 'Z', 'Y', 'Race', 'FICO_star'].
    Race and FICO_star are audit-only; models use (FICO, LTV, DTI, Z) to predict Y.
    Race and FICO_star are independent by construction; only FICO is biased for Race A.
    """

    rng = np.random.default_rng(seed)

    # Race (audit-only)
    race_is_a = rng.binomial(1, 0.35, size=n_samples)
    race_labels = np.where(race_is_a == 1, "A", "B")

    # ZIP proxy correlated with race
    p_z = np.where(race_is_a == 1, 0.60, 0.20)
    z = rng.binomial(1, p_z)

    # Latent true credit score, independent of race
    fico_star = rng.normal(loc=720.0, scale=60.0, size=n_samples)

    # Observed FICO with measurement bias for race A
    eps = rng.normal(loc=0.0, scale=25.0, size=n_samples)
    fico = fico_star - 35.0 * race_is_a + eps

    # Financial ratios (independent of race)
    ltv = _truncated_normal(rng, mean=0.80, std=0.10, lower=0.50, upper=1.10, size=n_samples)
    dti = _truncated_normal(rng, mean=0.35, std=0.12, lower=0.05, upper=0.80, size=n_samples)

    # Default probability driven by latent FICO
    alpha_f, alpha_l, alpha_d = -0.015, -3.0, -2.0
    eta_no_intercept = alpha_f * fico_star - alpha_l * ltv - alpha_d * dti
    alpha_0 = _calibrate_intercept(eta_no_intercept, target_rate=0.20, lower=-10.0, upper=10.0, n_iter=50)
    p_default = _sigma(alpha_0 + eta_no_intercept)

    # Default outcome
    y = rng.binomial(1, p_default)

    return pd.DataFrame(
        {
            "FICO": fico,
            "LTV": ltv,
            "DTI": dti,
            "Z": z,
            "Y": y,
            "Race": race_labels,
            "FICO_star": fico_star,
        }
    )


def train_test_split_mortgage(
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
