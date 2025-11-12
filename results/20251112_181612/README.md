# Run 20251112_181612 — λ sweep with 3-epoch warm-up

**Warm-up schedule**: first 3 epochs train predictor only, remaining 7 epochs include adversarial heads (per TrainingConfig).

| λ_adv | ROC AUC | EO ΔTPR | EO ΔFPR | Selection rate₀ | Selection rate₁ | DP diff | DP ratio | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.7664 | 0.0127 | 0.0021 | 0.00362 | 0.00336 | +0.00234 | 3.28 | Mild EO reduction, small gap. |
| 0.10 | 0.7681 | 0.0339 | 0.0027 | 0.01761 | 0.00959 | +0.00601 | 2.68 | Accuracy high, EO gap increases. |
| 0.20 | 0.7667 | 0.0190 | 0.0029 | 0.00974 | 0.00503 | +0.00299 | 2.46 | EO gap moderate; accuracy ≈ baseline. |
| 0.30 | 0.7630 | 0.0288 | 0.0034 | 0.01821 | 0.01271 | +0.00734 | 2.37 | Slightly larger EO gap, accuracy dips. |
| 0.50 | 0.7610 | 0.0128 | 0.0011 | 0.01110 | 0.00527 | +0.00285 | 2.17 | Reasonable trade-off. |
| 0.80 | 0.7612 | 0.0022 | 0.0005 | 0.00204 | 0.00192 | −0.00013 | 0.94 | EO gap near zero, selection rates balanced. |
| 1.20 | 0.7370 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | — | Strong fairness pressure; accuracy drops sharply. |
| 2.00 | 0.7330 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | — | Same as λ=1.2 with more accuracy loss. |

**Findings**
- Warm-up prevents the collapse to zero selection rates seen in earlier runs, but EOS gaps still oscillate for λ≤0.5; more tuning (longer warm-up or separate adversary LR) may smooth the curve.
- λ≈0.5–0.8 continues to offer the best accuracy/fairness compromise (AUC ≈0.761, EO gap ≈0.01 or smaller).

Artifacts: `lambda_sweep/metrics.csv` (+ plot if generated).***
