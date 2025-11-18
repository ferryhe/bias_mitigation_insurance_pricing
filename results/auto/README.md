# Auto results directory

All auto product experiments are logged under `results/auto/<run-id>/`, where `<run-id>` is a timestamp (`YYYYMMDD_HHMMSS`).

Each run folder should contain:

- `README.md` - summary of goals, configs, and highlights.
- Core CSV artifacts such as `baseline_results.csv` (and future `lambda_sweep/`, `sanity_checks/`, etc.).
- Optional visuals (e.g., `fairness_accuracy_frontier.png`, `debug_auto_distributions/` histograms).

## Adding a new run

1. Run one or more scripts under `src/experiments/auto/` (`baseline`, `lambda_sweep`, `sanity_checks`, `fairness_frontier`, `bias_sweep`, or `full_pipeline`).
2. Move all outputs for that execution into a fresh `results/auto/<run-id>/` folder.
3. Document the run in `results/auto/<run-id>/README.md` with configs, metrics tables, and observations.
4. Append a short entry to the log below.

## Run log

| Run ID | Experiments | Description |
| --- | --- | --- |
| 20251113_203022 | baseline | Initial auto baseline rerun (2% acceptance slice, debug histograms) to align the auto product with the credit reporting structure. |
| 20251113_203720 | bias_sweep | Bias-strength sweep (`[0.0, 0.25, 0.5, 1.0, 2.0]`) to stress-test fairness gaps and log the sweep metrics in `bias_sweep_metrics.csv`. |
