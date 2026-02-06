from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _sigma(x: np.ndarray) -> np.ndarray:
    """Logistic link σ(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(values: np.ndarray) -> np.ndarray:
    """Standardize to mean 0, std 1 (credit-style helper)."""
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    mean = values.mean()
    return (values - mean) / std


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


def generate_life_underwriting_data(
    n_samples: int = 100_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate a life underwriting dataset whose *risk structure* is very close
    to the Credit DGP, but with life variable names:

      - Latent credit Z_star plays the role of S_star (true FICO).
      - Observed credit Z plays the role of S (biased FICO).
      - Territory T plays the role of ZIP proxy Z (credit).
      - A continuous BMI-based factor plays the role of D (DTI).
      - Risky occupation R plays the role of L (delinquencies).

    The true label Y is driven mainly by Z_star plus these two risk factors,
    as in credit; Age/BMI/S/C remain in the dataset for realism and EDA.
    """

    rng = np.random.default_rng(seed)

    # 1. Race and territory proxy T (same as life slides)
    race_is_a = rng.binomial(1, 0.35, size=n_samples)
    p_t = np.where(race_is_a == 1, 0.60, 0.20)
    territory = rng.binomial(1, p_t)
    race_labels = np.where(race_is_a == 1, "A", "B")

    # 2. Latent, race-neutral credit Z_star (like S_star in credit)
    #    Use Normal instead of TruncNorm to be closer to credit.
    z_star = rng.normal(loc=700.0, scale=50.0, size=n_samples)

    # 3. Observed credit Z with measurement bias against Race A
    #    (like S = S_star - b * 1{Race=A} + noise in credit)
    b = 40.0
    tau_z = 25.0
    eps_z = rng.normal(loc=0.0, scale=tau_z, size=n_samples)
    z_obs = z_star - b * race_is_a + eps_z
    z_obs = np.clip(z_obs, 300.0, 850.0)

    # 4. Age & BMI (life-style, independent of Race except via BMI_cat tuning)
    age = _truncated_normal(
        rng=rng,
        mean=45.0,
        std=15.0,
        lower=20.0,
        upper=75.0,
        size=n_samples,
    )
    age_g = (age - 45.0) / 10.0

    # BMI categories correlated with Race (per reviewer: Race A heavier)
    u = rng.random(n_samples)
    bmi_cat = np.empty(n_samples, dtype=int)

    probs_a = np.array([0.35, 0.375, 0.275])  # Race A: more overweight/obese
    thresh_a = np.cumsum(probs_a)
    probs_b = np.array([0.40, 0.35, 0.25])  # Race B: more even
    thresh_b = np.cumsum(probs_b)

    mask_a = race_is_a == 1
    mask_b = ~mask_a
    u_a = u[mask_a]
    u_b = u[mask_b]

    bmi_cat[mask_a] = np.select(
        [u_a < thresh_a[0], u_a < thresh_a[1]],
        [0, 1],
        default=2,
    )
    bmi_cat[mask_b] = np.select(
        [u_b < thresh_b[0], u_b < thresh_b[1]],
        [0, 1],
        default=2,
    )

    bmi = np.empty(n_samples, dtype=float)
    for cat, mean, std, lower, upper in [
        (0, 22.0, 2.5, 16.0, 25.0),  # normal
        (1, 27.5, 2.5, 25.0, 30.0),  # overweight
        (2, 32.0, 3.0, 30.0, 45.0),  # obese
    ]:
        mask = bmi_cat == cat
        n_cat = int(mask.sum())
        if n_cat > 0:
            bmi[mask] = _truncated_normal(
                rng=rng,
                mean=mean,
                std=std,
                lower=lower,
                upper=upper,
                size=n_cat,
            )

    # 5. Lifestyle flags S (smoker), C (chronic), R (risky job)
    #    Keep simple life-style logic; these do NOT need to match credit.
    s_0 = -1.10
    s_age = 0.10
    p_s = _sigma(s_0 + s_age * age_g)
    s_smoker = rng.binomial(1, p_s)

    c_0 = -1.75
    c_bmi = 0.40
    c_age = 0.35
    bmi_g = (bmi - 27.0) / 5.0
    p_c = _sigma(c_0 + c_bmi * bmi_g + c_age * age_g)
    c_chronic = rng.binomial(1, p_c)

    # Risky occupation: a bit higher overall than original to mimic "L"
    base_r = 0.10
    uplift_r_t = 0.10
    p_r = base_r + uplift_r_t * territory   # 0.10 in T=0, 0.20 in T=1
    p_r = np.clip(p_r, 0.01, 0.99)
    r_risky = rng.binomial(1, p_r, size=n_samples)

    # 6. Territory-dominant risk structure:
    #    Make T and its interactions with Age/BMI very strong,
    #    and Z_star only a modest contributor.

    # (age_g and bmi_g already defined above)
    # age_g = (age - 45.0) / 10.0
    # bmi_g = (bmi - 27.0) / 5.0

    # Main effects
    beta_age =  0.10   # weak positive with age
    beta_bmi =  0.10   # weak positive with BMI
    beta_s   =  0.40   # smoker more risky
    beta_c   =  0.80   # chronic more risky
    beta_r   =  0.40   # risky job more risky
    gamma_z  =  -1.50   # higher Z_star lowers risk (stronger than original)

    # Nonlinear and interaction effects to foil simple GLMs
    beta_age2     =  0.05
    beta_bmi2     =  0.06
    beta_age_bmi  =  0.08
    beta_s_age    =  0.15
    beta_c_bmi    =  0.20
    gamma_z_quad  =  0.12

    # Linear predictor without intercept
    eta_no_intercept = (
        beta_age * age_g
        + beta_bmi * bmi_g
        + beta_s * s_smoker
        + beta_c * c_chronic
        + beta_r * r_risky
        + gamma_z * (z_star - 700.0) / 50.0
        + beta_age2 * age_g**2
        + beta_bmi2 * bmi_g**2
        + beta_age_bmi * age_g * bmi_g
        + beta_s_age * s_smoker * age_g
        + beta_c_bmi * c_chronic * bmi_g
        + gamma_z_quad * ((z_star - 700.0) / 50.0) ** 2
    )

    # Target event rate, e.g. 15%
    beta_0 = _calibrate_intercept(eta_no_intercept, target_rate=0.15)
    p_y = _sigma(beta_0 + eta_no_intercept)
    y = rng.binomial(1, p_y)


    df = pd.DataFrame(
        {
            "Age": age,
            "BMI": bmi,
            "S": s_smoker,
            "C": c_chronic,
            "R": r_risky,
            "Z": z_obs,
            "T": territory,
            "Y": y,
            "Race": race_labels,
            "Z_star": z_star,
            "BMI_cat": bmi_cat,
        }
    )
    return df



def train_test_split_life(
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
