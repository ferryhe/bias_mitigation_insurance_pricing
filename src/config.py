from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationConfig:
    n_samples: int = 60_000
    seed: int = 42
    p_raceA: float = 0.35
    p_z_given_raceA: float = 0.65
    p_z_given_raceB: float = 0.25
    mu_s: float = 700.0
    sigma_s: float = 60.0
    bias_b: float = 35.0
    tau_s: float = 25.0
    mu_d: float = -0.9
    sigma_d: float = 0.45
    lambda_l: float = 0.30
    alpha_s: float = -1.20
    alpha_d: float = 1.25
    alpha_l: float = 0.60
    target_default_rate: float = 0.12


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 1_024
    n_epochs_nn: int = 10
    n_epochs_adv: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_adv: float = 0.1
    warmup_epochs_adv: int = 5


@dataclass(slots=True)
class EvalConfig:
    threshold: float = 0.5
    target_acceptance_rate: float | None = None


def get_default_configs() -> tuple[SimulationConfig, TrainingConfig, EvalConfig]:
    """
    Returns the default (simulation, training, evaluation) configurations.
    """

    return SimulationConfig(), TrainingConfig(), EvalConfig()
