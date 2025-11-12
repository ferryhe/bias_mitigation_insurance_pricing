# Run 20251112_183549 — Fairness–accuracy frontier plot

**Inputs**
- λ sweep metrics (copied from `results/20251112_182507/lambda_sweep/metrics.csv`).
- Fixed-rate comparison metrics (from `results/20251112_183201/fixed_rate/metrics.csv`).
- Plot saved as `fairness_frontier/fairness_accuracy_frontier.png`.

**Plot description**
- X-axis: EO TPR gap, Y-axis: ROC AUC.
- Blue dots: adversarial λ sweep points, each annotated with its λ value.
- Orange diamond: GLM baseline (2% approval rate), Green diamond: plain NN baseline.
- Shows how adversarial models trace a curve from high AUC/high EO gap (low λ) toward low EO gap/lower AUC (high λ), while GLM/NN anchor the baseline quadrant.

**Observations**
- Points cluster around EO gaps ≤0.03 except for λ≈0.05–0.3, reinforcing that mild λ already achieves significant EO control.
- GLM sits at higher AUC but larger EO gap (>0.07); ADV_NN λ≈0.8 lands close to GLM’s accuracy with much better EO gap (<0.01).
- The plot makes it easy to reason about policy choices: pick λ that places you on the desired fairness–accuracy frontier.
