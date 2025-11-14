# Run 20251113_203022 - Auto baseline (2% acceptance)

**Purpose**
- Recreate the initial auto baseline so the auto product line mirrors the documented credit workflow.
- Capture a clean metrics snapshot plus histogram diagnostics in `debug_auto_distributions/` for future comparisons.

**Configuration**
- Simulator: default `AutoSimulationConfig` (60k rows, seed 202, `bias_strength = 1.0`, `p_protected = 0.30`).
- Training: default `TrainingConfig` (batch 1024, 10 epochs with a 5-epoch predictor warm-up, `lambda_adv = 0.1`), device = CPU.
- Evaluation: threshold 0.5 plus a fixed 2% acceptance-rate slice (`target_acceptance_rate = 0.02`).

---

## Baseline metrics (`baseline_results.csv`)

| Model | ROC AUC | PR AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
| --- | --- | --- | --- | --- | --- | --- |
| GLM | 0.757 | 0.902 | 0.067 | 0.155 | +0.097 | 1.11 |
| NN | 0.776 | 0.913 | 0.001 | 0.008 | +0.003 | 1.00 |
| ADV_NN | 0.811 | 0.928 | 0.089 | 0.264 | +0.152 | 1.19 |

- ADV_NN delivers the strongest accuracy but still exhibits EO/DP gaps at the default 0.5 threshold because the mitigation only starts after the warm-up phase.
- The plain NN nearly saturates TPR and FPR at 1.0 due to the highly imbalanced targets; it needs tighter thresholds to expose fairness differences.
- GLM lags in accuracy and keeps sizable DP and EO gaps, making it the least preferred baseline for auto.

---

## Fixed 2% acceptance snapshot

| Model | EO ΔTPR (2%) | EO ΔFPR (2%) | DP diff (2%) | DP ratio (2%) |
| --- | --- | --- | --- | --- |
| GLM | 0.048 | 0.000 | +0.040 | 5.83 |
| NN | 0.041 | 0.000 | +0.034 | 4.45 |
| ADV_NN | 0.030 | 0.000 | +0.026 | 3.05 |

- Forcing equal acceptance budgets clarifies the fairness trade-offs: ADV_NN keeps the lowest EO gap and DP skew among the three, though DP ratios remain high because approvals are extremely scarce.
- GLM continues to over-index on the protected class even at the constrained rate, suggesting calibration or proxy removal alone will not close DP gaps.
- Follow-up: extend this run with lambda sweeps and proxy/bias sanity checks (mirroring `results/credit/20251112_175008`) now that the baseline is logged.
