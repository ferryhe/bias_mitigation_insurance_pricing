# Results directory

Each complete experiment run is stored under `results/<run-id>/`, where `<run-id>` is a timestamp (`YYYYMMDD_HHMMSS`).  
Inside each run folder:

- `README.md` — consolidated summary (settings, key findings, next steps).
- Subfolders per experiment (`baseline/`, `lambda_sweep/`, `sanity_bias_vs_nobias/`, `sanity_with_vs_without_proxy/`, …), each containing `metrics.csv` (plus optional artifacts like plots).

To add a new run:

1. Create `results/<run-id>/` after executing the scripts.
2. Move each experiment’s outputs into subfolders inside that run directory.
3. Update `results/<run-id>/README.md` with metrics tables and commentary.
4. Append the run to the log below.

| Run ID | Experiments | Description |
| --- | --- | --- |
| 20251112_175008 | baseline, lambda_sweep, sanity_bias_vs_nobias, sanity_with_vs_without_proxy | Default configs after ADV_NN fix (`lambda_adv = 0.1`). |
| 20251112_175645 | lambda_sweep | λ sweep after removing double-counting of `lambda_adv` in adversarial training. |
| 20251112_180036 | baseline | Added acceptance-rate thresholding (r = 2%) to compare fairness at matched selection rates. |
| 20251112_181121 | lambda_sweep | Fine-grained λ grid `[0.05 … 2.0]` with detailed fairness columns. |
| 20251112_181612 | lambda_sweep | Same grid, now with 3-epoch warm-up (predictor-only) before adversarial training. |
| 20251112_182507 | lambda_sweep | Warm-up increased to 5 epochs; evaluates stability vs accuracy trade-offs. |
| 20251112_183201 | fixed_rate | Compares GLM/NN/ADV_NN at a shared 2% approval rate. |
| 20251112_183549 | fairness_frontier | Scatter plot of ROC AUC vs EO gap (λ sweep + GLM/NN baselines). |
| 20251112_184018 | fixed_rate | Extends comparison to 1%, 2%, 5% approval rates. |
| 20251112_184338 | fairness_vs_rate | Line plots of DP ratio and EO gap vs approval rate. |
