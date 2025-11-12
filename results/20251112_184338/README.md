# Run 20251112_184338 — Fairness vs approval rate plots

**Inputs**
- Fixed-rate metrics from `results/20251112_184018/fixed_rate/metrics.csv`.

**Artifacts**
- `fairness_vs_rate/dp_ratio.png`: DP ratio vs approval rate (1%, 2%, 5%) for GLM, NN, ADV_NN.
- `fairness_vs_rate/eo_gap.png`: EO TPR gap vs approval rate.

**Observations**
- DP ratio decreases for all models as the approval rate increases; ADV_NN stays closest to parity across rates (DP ratio ≈1–1.3).
- EO gaps also shrink with higher approval rates, with ADV_NN consistently the lowest.
- These plots highlight that ADV_NN provides the largest fairness improvement at tight approval budgets, while differences narrow as approval rates rise.
