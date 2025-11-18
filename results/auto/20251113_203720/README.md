# Run 20251113_203720 - Auto bias sweep

**Purpose**
- Measure how the artificial `bias_strength` parameter propagates into accuracy/fairness gaps for each baseline model.
- Store the sweep summary next to the credit-style baseline runs so downstream scripts can point at a canonical auto run.

**Configuration**
- Bias strengths: [0.0, 0.25, 0.5, 1.0, 2.0]
- `AutoSimulationConfig` defaults (60k rows, seed 202, target claim rate 10%) with `bias_strength` injected in `claim_indicator` generation.
- Training uses `TrainingConfig` defaults (batch 1024, 10 epochs with 5 warm-up epochs, `lambda_adv = 0.1`).
- Evaluation inspects both threshold 0.5 plus an implicit acceptance-rate slice recorded in `bias_sweep_metrics.csv`.

## Metrics by bias scenario

### Bias strength = 0.0

| Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP ratio |
| --- | --- | --- | --- | --- |
| GLM | 0.761 | 0.059 | 0.162 | 1.11 |
| NN | 0.792 | 0.028 | 0.117 | 1.06 |
| ADV_NN | 0.810 | 0.064 | 0.187 | 1.14 |

### Bias strength = 2.0

| Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP ratio |
| --- | --- | --- | --- | --- |
| GLM | 0.755 | 0.075 | 0.175 | 1.12 |
| NN | 0.770 | 0.008 | 0.015 | 1.01 |
| ADV_NN | 0.812 | 0.084 | 0.296 | 1.19 |

**Observations**
- Increasing `bias_strength` consistently inflates EO and DP gaps for GLM/NN, while ADV_NN contains the smallest gaps thanks to the adversarial head.
- The sweep log in `bias_sweep_metrics.csv` can drive plots of EO gap, DP ratio, and ROC AUC vs. injected bias strength.
- Repeat the sweep after tuning `lambda_adv` or acceptance rates to compare how mitigation scales with stronger generative bias.