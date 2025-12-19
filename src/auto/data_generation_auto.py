from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from src.config import AutoSimulationConfig


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


def _as_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1e-6, 1.0 - 1e-6)


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


def generate_auto_insurance_data(
    sim_cfg: AutoSimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate an auto insurance dataset aligned with AutoSimulationConfig.

    Returns columns expected by the auto experiments:
    ['territory','age','years_licensed','annual_mileage','vehicle_use',
     'vehicle_age','vehicle_value','safety_score','past_claims_obs',
     'violations_obs','credit_score','income','claim_indicator','A'].
    """

    sim_cfg = sim_cfg or AutoSimulationConfig()
    rng = np.random.default_rng(sim_cfg.seed)
    n = sim_cfg.n_samples

    A = rng.binomial(1, sim_cfg.p_protected, size=n)
    territory_p = _as_probability(0.55 + 0.20 * A)
    territory = rng.binomial(1, territory_p)

    age = np.clip(rng.normal(45.0 - 2.0 * A, 12.0, size=n), 18.0, 85.0)
    years_licensed = np.clip(age - rng.normal(17.0, 2.0, size=n), 0.0, None)
    annual_mileage = np.clip(rng.normal(12_000.0, 3_000.0, size=n), 4_000.0, 35_000.0)
    vehicle_use = rng.binomial(1, _as_probability(0.40 + 0.10 * A))
    vehicle_age = np.clip(rng.exponential(6.0, size=n), 0.0, 25.0)
    vehicle_value = np.clip(
        rng.lognormal(mean=np.log(22_000.0) - 0.05 * A, sigma=0.35, size=n),
        5_000.0,
        80_000.0,
    )
    safety_score = np.clip(rng.normal(0.70 - 0.05 * A, 0.10, size=n), 0.20, 0.95)

    lambda_claims = np.clip(0.20 + 0.30 * territory + 0.10 * A, 0.01, None)
    past_claims_true = rng.poisson(lam=lambda_claims, size=n)
    lambda_violations = np.clip(0.40 + 0.30 * territory + 0.05 * (annual_mileage / 1_000.0), 0.01, None)
    violations_true = rng.poisson(lam=lambda_violations, size=n)

    extra_claims = rng.binomial(1, _as_probability(sim_cfg.p_extra_claim * A), size=n)
    extra_violations = rng.binomial(1, _as_probability(sim_cfg.p_extra_violation * A), size=n)
    past_claims_obs = past_claims_true + extra_claims
    violations_obs = violations_true + extra_violations

    credit_score = np.clip(rng.normal(700.0 - 30.0 * A - 15.0 * territory, 55.0, size=n), 500.0, 850.0)
    income = np.clip(
        rng.lognormal(mean=np.log(70_000.0) - 0.12 * A, sigma=0.45, size=n),
        20_000.0,
        250_000.0,
    )

    exposure = 1.0 + 0.2 * vehicle_use + 0.05 * territory
    mileage_scaled = annual_mileage / 10_000.0
    value_log = np.log1p(vehicle_value / 10_000.0)
    safety_sq = safety_score ** 2
    age_sq = (age / 10.0) ** 2
    age_cu = (age / 10.0) ** 3

    logit = (
        sim_cfg.beta_age2 * age_sq
        + 0.08 * age_cu
        + sim_cfg.beta_exp * (years_licensed / 10.0)
        + sim_cfg.beta_mileage * mileage_scaled
        + 0.35 * mileage_scaled ** 2
        + sim_cfg.beta_territory * territory
        + sim_cfg.beta_use * vehicle_use
        + sim_cfg.beta_past_claims * np.log1p(past_claims_true)
        + sim_cfg.beta_violations * np.log1p(violations_true)
        + 0.30 * mileage_scaled * safety_score
        + 0.25 * value_log * territory
        + 0.20 * np.log1p(violations_true) * safety_sq
        + 0.12 * (age / 10.0) * (vehicle_age / 10.0)
        + 0.10 * value_log * mileage_scaled
        + 0.25 * exposure
    )
    logit += sim_cfg.bias_strength * (0.5 * territory + 0.8 * A)

    beta_0 = _calibrate_intercept(logit, target_rate=sim_cfg.target_claim_freq)
    p_claim = _sigma(beta_0 + logit)
    claim_indicator = rng.binomial(1, p_claim)

    df = pd.DataFrame(
        {
            "A": A,
            "territory": territory,
            "age": age,
            "years_licensed": years_licensed,
            "annual_mileage": annual_mileage,
            "vehicle_use": vehicle_use,
            "vehicle_age": vehicle_age,
            "vehicle_value": vehicle_value,
            "safety_score": safety_score,
            "past_claims_obs": past_claims_obs,
            "violations_obs": violations_obs,
            "credit_score": credit_score,
            "income": income,
            "claim_indicator": claim_indicator,
            "past_claims_true": past_claims_true,
            "violations_true": violations_true,
        }
    )

    return df



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
