from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd


RESULTS_DIR = Path("results")


def _latest_run() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(
        (p for p in RESULTS_DIR.iterdir() if p.is_dir()),
        key=lambda x: x.name,
        reverse=True,
    )
    return runs[0] if runs else None


def _baseline_metrics(run_root: Path) -> pd.DataFrame | None:
    candidate = run_root / "baseline" / "metrics.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    return None


def _summary_text(run_root: Path) -> str:
    readme = run_root / "README.md"
    if readme.exists():
        return readme.read_text()
    return f"Detailed README not found in {run_root}"


def _plot_images() -> list[str]:
    files = [
        RESULTS_DIR / "fairness_accuracy_frontier.png",
        RESULTS_DIR / "fairness_vs_rate_dp.png",
        RESULTS_DIR / "fairness_vs_rate_eo.png",
    ]
    return [str(p) for p in files if p.exists()]


def load_latest_run():
    run_root = _latest_run()
    if run_root is None:
        return (
            "No runs found",
            "Run an experiment (e.g., `python -m src.experiments.run_baseline`) to generate data.",
            pd.DataFrame(),
            [],
        )

    df = _baseline_metrics(run_root)
    images = _plot_images()
    summary = _summary_text(run_root)
    return run_root.name, summary, df if df is not None else pd.DataFrame(), images


with gr.Blocks(title="Credit Fairness Dashboard") as demo:
    gr.Markdown("# Credit Insurance Fairness Study")
    gr.Markdown("Latest run artifacts under `results/`. Metrics and plots (frontier, DP/EO vs rate) are collected below.")

    with gr.Row():
        run_name = gr.Textbox(label="Latest run", interactive=False)
        summary_md = gr.Markdown("")

    splitter = gr.Box()
    baseline_table = gr.Dataframe(
        value=pd.DataFrame(), label="Baseline metrics (GLM / NN / ADV_NN)"
    )

    gallery = gr.Gallery(label="Fairness plots", columns=3).style(height="240px")

    def refresh():
        name, summary, df, imgs = load_latest_run()
        return name, summary, df, imgs

    refresh_btn = gr.Button("Refresh latest run")
    refresh_btn.click(
        refresh,
        inputs=[],
        outputs=[run_name, summary_md, baseline_table, gallery],
        queue=False,
    )

    refresh()

if __name__ == "__main__":
    demo.launch()
