from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_default_configs
from src.data_generation import generate_credit_insurance_data, train_test_split_df
from src.evaluation.fairness import fairness_at_target_rate
from src.evaluation.metrics import compute_accuracy_metrics
from src.models.adv_nn_model import AdvPredictor, predict_proba_adv_nn, train_adv_nn
from src.models.glm_model import GLMClassifier
from src.models.nn_model import PlainNN, predict_proba_plain_nn, train_plain_nn


TARGET_RATES = [0.01, 0.02, 0.05]
ADV_LAMBDA = 0.8


def _prepare_features(df, scaler: StandardScaler | None = None):
    numeric_cols = ["S", "D", "L"]
    proxy_col = "Z"
    numeric = df[numeric_cols].to_numpy(dtype=np.float32)
    proxy = df[[proxy_col]].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric)
    else:
        numeric_scaled = scaler.transform(numeric)

    X = np.concatenate([numeric_scaled, proxy], axis=1)
    y = df["Y"].to_numpy(dtype=np.float32)
    A = df["A"].to_numpy(dtype=np.int64)
    return X, y, A, scaler


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_loader_with_A(
    X: np.ndarray, y: np.ndarray, A: np.ndarray, batch_size: int
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(A).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _evaluate_model(model_name: str, y_test, y_proba, A_test, target_rate: float):
    accuracy = compute_accuracy_metrics(y_test, y_proba)
    fairness = fairness_at_target_rate(y_test, y_proba, A_test, target_rate)
    return {
        "model_name": model_name,
        "target_rate": target_rate,
        **accuracy,
        **fairness,
    }


def main() -> None:
    sim_cfg, train_cfg, eval_cfg = get_default_configs()

    df = generate_credit_insurance_data(sim_cfg)
    df_train, df_test = train_test_split_df(df, test_size=0.2, seed=sim_cfg.seed)

    X_train, y_train, A_train, scaler = _prepare_features(df_train, None)
    X_test, y_test, A_test, _ = _prepare_features(df_test, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GLM
    glm = GLMClassifier()
    glm.fit(X_train, y_train)
    y_proba_glm = glm.predict_proba(X_test)

    # Plain NN
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=sim_cfg.seed,
        stratify=y_train,
    )

    train_loader = _build_loader(X_tr, y_tr, train_cfg.batch_size, shuffle=True)
    val_loader = _build_loader(X_val, y_val, train_cfg.batch_size, shuffle=False)
    plain_nn = PlainNN(input_dim=X_train.shape[1]).to(device)
    train_plain_nn(plain_nn, train_loader, val_loader, train_cfg, device)
    y_proba_nn = predict_proba_plain_nn(plain_nn, X_test, device=device)

    # Adversarial NN
    train_cfg_adv = replace(train_cfg, lambda_adv=ADV_LAMBDA)
    adv_loader = _build_loader_with_A(
        X_train, y_train, A_train, train_cfg.batch_size
    )
    adv_model = AdvPredictor(input_dim=X_train.shape[1]).to(device)
    train_adv_nn(adv_model, adv_loader, train_cfg_adv, device=device)
    y_proba_adv = predict_proba_adv_nn(adv_model, X_test, device=device)

    rows: list[dict] = []
    for target_rate in TARGET_RATES:
        rows.append(_evaluate_model("GLM", y_test, y_proba_glm, A_test, target_rate))
        rows.append(_evaluate_model("NN", y_test, y_proba_nn, A_test, target_rate))
        rows.append(
            _evaluate_model("ADV_NN", y_test, y_proba_adv, A_test, target_rate)
        )

    df_results = pd.DataFrame(rows)
    print(df_results)

    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/fixed_rate_comparison.csv", index=False)


if __name__ == "__main__":
    main()
