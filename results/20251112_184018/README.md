# Run 20251112_184018 — Fixed-rate comparison (1%, 2%, 5%)

**Setup**
- Approval rates: 1%, 2%, 5% (global top-scoring applicants).
- Models: GLM, Plain NN, ADV_NN (`λ = 0.8`, 5 warm-up epochs).
- Dataset: default simulation (60 k rows, bias_b = 35, seed 42).

| Target rate | Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1% | GLM | 0.772 | 0.052 | 0.0074 | +0.012 | 3.13 |
| 1% | NN | 0.751 | 0.058 | 0.0082 | +0.0137 | 3.62 |
| 1% | ADV_NN | 0.752 | 0.0119 | 0.0021 | +0.00011 | 1.01 |
| 2% | GLM | 0.772 | 0.072 | 0.016 | +0.022 | 2.77 |
| 2% | NN | 0.751 | 0.070 | 0.011 | +0.017 | 2.67 |
| 2% | ADV_NN | 0.752 | 0.066 | 0.015 | +0.020 | 1.27 |
| 5% | GLM | 0.772 | 0.060 | 0.012 | +0.0218 | 2.22 |
| 5% | NN | 0.751 | 0.040 | 0.006 | +0.017 | 1.77 |
| 5% | ADV_NN | 0.752 | 0.034 | 0.004 | +0.009 | 1.10 |

**Observations**
- Lower approval rates amplify disparities for GLM/NN (DP ratio >3 at 1%), while ADV_NN keeps DP ratio ≈1 and EO gaps ~0.01.
- At higher rates (5%), fairness gaps shrink for all models, though ADV_NN still maintains slightly better EO/DP than GLM/NN.
- Accuracy remains similar across rates (GLM highest, NN ≈ ADV_NN).

Artifacts: `fixed_rate/metrics.csv`.
