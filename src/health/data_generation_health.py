from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_HEALTH_PARAMS: Dict[str, float] = {
    # Demographics / proxy
    "p_raceA": 0.25,
    "p_female": 0.50,
    "p_adi_given_A": 0.55,
    "p_adi_given_B": 0.25,
    "age_mean": 45.0,
    "age_sd": 15.0,
    "age_min": 18.0,
    "age_max": 85.0,
    "age_scale": 10.0,
    # Chronic probability (race-neutral)
    "alpha0": -0.5,
    "alpha_age": 0.45,
    "alpha_fem": 0.1,
    "alpha_u": 0.4,
    # Latent morbidity M_star (simplified)
    "theta0": 0.0,
    "theta_u": 0.9,
    "theta_age": 0.35,
    "theta_c": 0.8,
    "sigma_mstar": 0.45,
    # Measurement bias (m_obs)
    "b_meas": 0.35,  # Race A appears sicker
    "sigma_meas": 0.35,
    # Outcome model (solved intercept to hit target rate)
    "beta_m": 1.0,
    "beta_age": 0.15,
    "beta_c": 0.25,
    "target_event_rate": 0.12,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic link."""
    return 1.0 / (1.0 + np.exp(-x))


def truncated_normal(
    size: int,
    loc: float,
    scale: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simple rejection sampler for a truncated normal distribution.
    """

    if lower >= upper:
        raise ValueError("lower must be less than upper for truncated normal sampling.")

    samples = np.empty(int(size), dtype=float)
    filled = 0
    while filled < size:
        remaining = size - filled
        draws = rng.normal(loc=loc, scale=scale, size=remaining * 2)
        draws = draws[(draws >= lower) & (draws <= upper)]
        take = min(draws.size, remaining)
        if take == 0:
            continue
        samples[filled : filled + take] = draws[:take]
        filled += take
    return samples


def _standardize(values: np.ndarray) -> np.ndarray:
    """Center and scale; return zeros if variance is zero."""
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    mean = values.mean()
    return (values - mean) / std


def _solve_intercept(
    target_rate: float,
    m_std: np.ndarray,
    age_scaled: np.ndarray,
    chronic: np.ndarray,
    beta_m: float,
    beta_age: float,
    beta_c: float,
) -> tuple[float, np.ndarray]:
    """
    Binary search on the intercept to hit the target event rate (credit-style).
    """

    lower, upper = -8.0, 8.0
    intercept = 0.0
    p = np.zeros_like(m_std)
    for _ in range(80):
        intercept = 0.5 * (lower + upper)
        logits = (
            intercept
            + beta_m * m_std
            + beta_age * age_scaled
            + beta_c * chronic
        )
        p = sigmoid(logits)
        mean_p = p.mean()
        if abs(mean_p - target_rate) < 1e-4:
            break
        if mean_p > target_rate:
            upper = intercept
        else:
            lower = intercept
    return intercept, p


def simulate_health(
    n: int = 100_000, seed: int = 123, params: Dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Simulate a health underwriting dataset with a biased morbidity measurement.

    The true outcome y is race-neutral given the latent morbidity m_star.
    The observed morbidity m_obs is shifted upward for Race A, inducing
    measurement bias in downstream models.
    """

    p = {**DEFAULT_HEALTH_PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)

    # Protected attribute and proxy
    gA = rng.binomial(1, p["p_raceA"], size=n)
    race_group = np.where(gA == 1, "A", "B")
    p_adi = np.where(gA == 1, p["p_adi_given_A"], p["p_adi_given_B"])
    adi_high = rng.binomial(1, p_adi)

    # Demographics and latent frailty
    age = truncated_normal(
        size=n,
        loc=p["age_mean"],
        scale=p["age_sd"],
        lower=p["age_min"],
        upper=p["age_max"],
        rng=rng,
    )
    # keep continuous ages to avoid spiky integer bins
    age_scaled = (age - p["age_mean"]) / p["age_scale"]
    female = rng.binomial(1, p["p_female"], size=n)
    u = rng.normal(0.0, 1.0, size=n)

    # Race-neutral chronic condition
    p_chronic = sigmoid(
        p["alpha0"]
        + p["alpha_age"] * age_scaled
        + p["alpha_fem"] * (female - 0.5)
        + p["alpha_u"] * u
    )
    chronic = rng.binomial(1, p_chronic)

    # Latent true morbidity (simplified, race-neutral)
    m_star = (
        p["theta0"]
        + p["theta_u"] * u
        + p["theta_age"] * age_scaled
        + p["theta_c"] * chronic
        + rng.normal(0.0, p["sigma_mstar"], size=n)
    )

    # Observed morbidity score with measurement bias for Race A
    m_obs = m_star + p["b_meas"] * gA + rng.normal(
        0.0, p["sigma_meas"], size=n
    )

    # Objective hospitalization outcome (credit-style: intercept solved to target rate)
    m_std = _standardize(m_star)
    beta0, p_y = _solve_intercept(
        target_rate=p["target_event_rate"],
        m_std=m_std,
        age_scaled=age_scaled,
        chronic=chronic,
        beta_m=p["beta_m"],
        beta_age=p["beta_age"],
        beta_c=p["beta_c"],
    )
    y = rng.binomial(1, p_y)

    return pd.DataFrame(
        {
            "race_group": race_group,
            "adi_high": adi_high,
            "age": age,
            "age_scaled": age_scaled,
            "female": female,
            "chronic": chronic,
            "u": u,
            "m_star": m_star,
            "m_obs": m_obs,
            "p_y": p_y,
            "y": y,
        }
    )


def generate_health_underwriting_data(
    n_samples: int = 100_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper used by the demo notebook.
    """

    return simulate_health(n=n_samples, seed=seed)


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


# Alias to maintain compatibility with existing imports.
simulate_health_insurance_data = simulate_health
