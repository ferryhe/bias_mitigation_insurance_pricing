<div align="center">

# Credit Insurance Fairness Study

*Synthetic credit underwriting pipeline for benchmarking accuracy–fairness trade-offs.*

</div>

---

## 1. Project Overview

We simulate a credit-insurance setting where the *true* default risk is race-neutral, yet the **observed** credit score is biased against the protected group. Even without using race explicitly, this bias plus a correlated proxy (ZIP code) causes fairness violations.  

We compare three families of models:

1. **GLM (Logistic Regression)** – fast, interpretable baseline.
2. **Plain Neural Network** – small MLP that can exploit nonlinear interactions.
3. **Adversarial Neural Network (ADV_NN)** – predictor + adversary heads with Gradient Reversal to enforce Equalized Odds (EO).

The repository includes:
- Reproducible data generation (60k samples by default).
- Training scripts for all models (PyTorch + scikit-learn).
- Evaluation utilities for accuracy + fairness metrics.
- Experiments covering baseline comparisons, λ sweeps, fixed approval-rate fairness, and plotting scripts.

---

## 2. Key Concepts

| Concept | Summary |
| --- | --- |
| **Measurement bias** | Observed score `S = S* - bias_b * A + noise` systematically lowers protected-group values, even though latent score `S*` is race neutral. |
| **Proxy variable `Z`** | Binary ZIP-like feature correlated with race; models that ingest `Z` leak group information inadvertently. |
| **Protected attribute `A`** | Binary race indicator (0 = group B, 1 = group A); used only for auditing fairness. |
| **Equalized Odds (EO)** | Requires TPR and FPR parity across groups at a chosen threshold. We report EO gaps: `|TPR_A - TPR_B|`, `|FPR_A - FPR_B|`. |
| **Demographic Parity (DP)** | Requires equal overall selection rates: `Pr(ŷ=1 | A=1)` ≈ `Pr(ŷ=1 | A=0)`. We report DP difference and ratio. |

---

## 3. Repository Structure

```text
credit_fairness_study/
├─ README.md
├─ requirements.txt / pyproject.toml
├─ app.py                             # Streamlit front-end for browsing results
├─ notebooks/
│   └─ credit_fairness_demo.ipynb     # interactive walkthrough
├─ src/
│   ├─ __init__.py
│   ├─ config.py                       # dataclasses for sim / train / eval configs
│   ├─ data_generation.py              # simulator + stratified split
│   ├─ models/
│   │   ├─ __init__.py
│   │   ├─ glm_model.py                # scikit-learn logistic regression wrapper
│   │   ├─ nn_model.py                 # plain PyTorch MLP + training helpers
│   │   └─ adv_nn_model.py             # predictor trunk + adversaries + gradient reversal warm-up
│   ├─ training/
│   │   ├─ __init__.py
│   │   ├─ train_glm.py                # end-to-end GLM pipeline (scaling, training, metrics)
│   │   ├─ train_nn.py                 # plain NN training loop with loaders + eval
│   │   └─ train_adv_nn.py             # ADV_NN training with warm-up + fairness metrics
│   ├─ evaluation/
│   │   ├─ __init__.py
│   │   ├─ metrics.py                  # ROC/PR/Brier/log-loss helpers
│   │   ├─ fairness.py                 # EO/DP metrics + target-rate utilities + thresholding
│   │   └─ reporting.py                # format/save/print tables
│   └─ experiments/
│       ├─ __init__.py
│       ├─ run_baseline.py             # GLM/NN/ADV_NN comparison
│       ├─ run_lambda_sweep.py         # sweep λ for ADV_NN
│       ├─ run_sanity_checks.py        # bias on/off + proxy ablation
│       ├─ run_fixed_rate_comparison.py# fairness at fixed approval rates
│       ├─ plot_fairness_accuracy_frontier.py # EO gap vs ROC AUC scatter
│       └─ plot_fairness_vs_rate.py    # DP ratio & EO gap vs approval rate
├─ results/                            # timestamped experiment outputs
│   ├─ README.md                       # index of run IDs and experiment summaries
│   └─ <run-id>/                       # e.g., results/20251112_184338/
│       ├─ <experiment>/metrics.csv    # raw metrics
│       ├─ ...plots...                 # plots per experiment
│       └─ README.md                   # findings & recommendations for that run
└─ tests/                              # pytest coverage for data/metrics/models
```

---

## 4. Getting Started

```bash
python --version    # ensure >= 3.10
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                   # optional smoke tests (≈5s)
```

To generate a fresh dataset and run the default comparison:

```bash
python -m src.experiments.run_baseline
```

CSV outputs land in `results/<run-id>/...`.

---

## 5. Running the Demo Notebook

Interactive notebooks:

- `notebooks/credit_fairness_demo.ipynb` – full end-to-end run (data → models → metrics).
- `notebooks/exploration.ipynb` – scratchpad for ad-hoc analysis / plotting.

1. Activate the virtual environment (`source .venv/bin/activate`).
2. Launch Jupyter or VS Code’s notebook UI.
3. The notebook walks through:
   - Data generation (inspect distributions, group-level metrics).
   - Training GLM / NN / ADV_NN inline.
   - Visual diagnostics (score histograms, EO gap vs threshold).
4. Modify cells to try alternative configs (e.g., change `bias_b`, `lambda_adv`, approval rates).

---

## 6. Running Key Experiments

All scripts are executable via `python -m src.experiments.<name>` and record artifacts in `results/`.

### Baseline (GLM / NN / ADV_NN)
```bash
python -m src.experiments.run_baseline
```
Outputs `results/<run-id>/baseline/metrics.csv` plus summary README.

### Lambda sweep (accuracy–fairness frontier)
```bash
python -m src.experiments.run_lambda_sweep
python -m src.experiments.plot_fairness_accuracy_frontier
```
- Sweep logs ROC AUC + EO gaps vs λ (0.05 → 2.0).
- Plot script produces `results/fairness_accuracy_frontier.png` (also copied into the latest run folder, e.g., `results/<run-id>/fairness_frontier/`).

### Fixed-rate fairness comparison (1%, 2%, 5% approvals)
```bash
python -m src.experiments.run_fixed_rate_comparison
python -m src.experiments.plot_fairness_vs_rate
```
- Generates `results/fixed_rate_comparison.csv` plus timestamped copies (e.g., `results/20251112_184018/fixed_rate/metrics.csv`).
- Produces line charts: `results/fairness_vs_rate_dp.png`, `results/fairness_vs_rate_eo.png` (also archived under `results/20251112_184338/fairness_vs_rate/`).

### Additional scripts
- `run_sanity_checks` – toggles measurement bias & proxy usage.
- `run_sanity_checks` results help verify the bias mechanism is the source of unfairness.

Each command writes CSV/plots under `results/` and adds a timestamped folder with a short summary (see `results/README.md` for the run log).

### Launching the Gradio dashboard

```bash
gradio app.py
```

- Shows the latest run name, README text, baseline metrics table, and fairness plots.
- Ideal for deploying to Hugging Face Spaces (the Space will automatically start `app.py`).

### Adding a new experiment run

1. Move the artifacts from the script(s) you ran into `results/<YYYYMMDD_HHMMSS>/<experiment>/` (e.g., `lambda_sweep`, `fixed_rate`).
2. In each run folder, add a `README.md` capturing:
   - Configuration knobs (`lambda_adv`, target rates, bias flags).
   - Key metrics (accuracy + fairness) in table form.
   - Observations, anomalies, or follow-up ideas.
3. Update `results/README.md` to log the new run ID and link to its README.
4. Use existing run folders as templates for phrasing and formatting.

---

## 7. Model Architecture

**Adversarial NN (Equalized Odds)**  

```
Input x = [S, D, L] ──► Predictor trunk ──► Shared representation h
                                │
                                │ (prediction head)
                                ▼
                             ŷ logits
                                │
                                │ (Gradient Reversal Layer, λ)
                                ▼
                    ┌──────── Adversary head (Y=0) ─► predict Z
                    │
Shared h ──[GRL]────┤
                    │
                    └──────── Adversary head (Y=1) ─► predict Z
```

- Predictor minimizes BCE on default prediction.
- Adversary heads (one per true label) try to recover proxy `Z`; Gradient Reversal makes the predictor *confound* them, encouraging EO parity.
- Warm-up epochs train the predictor alone before adversaries kick in (configurable via `warmup_epochs_adv`).

---

## 8. Summary of Findings

- **GLM**: Highest ROC AUC (~0.77–0.78) but largest EO/DP gaps under measurement bias (EO ΔTPR ≈ 0.06–0.08; DP ratio up to 3.4 when approval rate is tight).
- **Plain NN**: Similar accuracy, fairness improves slightly but inconsistently; can even worsen DP because it overfits proxies.
- **ADV_NN (λ ≈ 0.8)**: Best fairness/accuracy trade-off. ROC AUC remains ~0.76 while EO gaps drop below 0.02–0.03 and DP ratios hover near 1.0–1.3.
- **Fixed approval rates (1–5%)**: ADV_NN keeps DP ratios between 1.0–1.3 vs GLM’s 2.2–3.1; gains are most pronounced at strict approval rates (1%).

**Figure guide**

| Figure | Description |
| --- | --- |
| `results/fairness_accuracy_frontier.png` | Scatter: EO gap vs ROC AUC for each λ + GLM/NN markers. |
| `results/fairness_vs_rate/dp_ratio.png` | Lines: DP ratio vs approval rate (GLM/NN/ADV_NN). |
| `results/fairness_vs_rate/eo_gap.png` | Lines: EO TPR gap vs approval rate. |

---

## 9. Future Extensions

- **Calibration**: Add isotonic/Platt scaling and compare fairness metrics post-calibration.
- **Threshold sweeps**: Evaluate EO/DP across thresholds to select policy-driven operating points.
- **Bootstrap confidence intervals**: Quantify uncertainty in ROC AUC and fairness metrics.
- **Alternative mitigations**: Implement reweighting, reject-option classification, or group-specific thresholds for comparison.
- **More proxies / features**: Add new biased signals (e.g., employment history) to study multi-proxy effects.

---

For questions or contributions, open an issue or PR. Happy experimenting! 🧪
