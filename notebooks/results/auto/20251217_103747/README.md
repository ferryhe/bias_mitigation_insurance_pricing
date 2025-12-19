# Run 20251217_103747 - Auto bias sweep

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
| GLM | 0.929 | 0.012 | 0.031 | 0.96 |
| NN | 0.926 | 0.014 | 0.016 | 0.96 |
| ADV_NN | 0.927 | 0.012 | 0.012 | 0.96 |

### Bias strength = 2.0

| Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP ratio |
| --- | --- | --- | --- | --- |
| GLM | 0.929 | 0.034 | 0.077 | 1.02 |
| NN | 0.926 | 0.042 | 0.059 | 1.03 |
| ADV_NN | 0.927 | 0.038 | 0.059 | 1.03 |

**Observations**
- GLM: EO ΔTPR rises from 0.012 to 0.034; DP ratio rises from 0.957 to 1.023.
- NN: EO ΔTPR rises from 0.014 to 0.042; DP ratio rises from 0.958 to 1.025.
- ADV_NN: EO ΔTPR rises from 0.012 to 0.038; DP ratio rises from 0.963 to 1.028.
- The sweep log in `bias_sweep_metrics.csv` can drive plots of EO gap, DP ratio, and ROC AUC vs. injected bias strength.
- Repeat the sweep after tuning `lambda_adv` or acceptance rates to compare how mitigation scales with stronger generative bias.