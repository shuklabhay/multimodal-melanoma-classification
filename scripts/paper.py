from __future__ import annotations

import sys
from pathlib import Path

from typer import Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paper.ablations import (
    build_ablation_run_plan,
    load_ablation_spec,
    write_default_ablation_spec,
)
from src.paper.figures import build_figures
from src.paper.harvest import harvest_wandb_runs
from src.paper.tables import build_tables

app = Typer()


@app.command()
def harvest(
    out_dir: str = "outputs/paper",
    download_confusions: bool = True,
) -> None:
    paths = harvest_wandb_runs(out_dir, download_confusions=download_confusions)
    print(f"Baseline JSON: {paths.baseline_json}")
    print(f"Narrative JSON: {paths.narrative_json}")
    print(f"GPT-5 JSON: {paths.gpt5_json}")


@app.command()
def tables(
    input_dir: str = "outputs/paper",
    manifests_dir: str = "assets/manifests",
) -> None:
    out = build_tables(input_dir, manifests_dir)
    for k, v in out.items():
        print(f"{k}: {v}")


@app.command()
def figures(
    input_dir: str = "outputs/paper",
    output_dir: str = "assets/figures/paper",
) -> None:
    out = build_figures(input_dir, output_dir)
    for k, v in out.items():
        print(f"{k}: {v}")


@app.command()
def scaffold_ablation(
    config_path: str = "config/paper_ablation.yaml",
) -> None:
    out = write_default_ablation_spec(config_path)
    spec = load_ablation_spec(out)
    print(f"Wrote ablation spec: {out}")
    print(f"Spec name: {spec.raw.get('name')}")


@app.command()
def ablation_plan(
    config_path: str = "config/paper_ablation.yaml",
    out_json: str = "outputs/paper/ablation_run_plan.json",
) -> None:
    spec = load_ablation_spec(config_path)
    paths = build_ablation_run_plan(spec, out_json)
    print(f"Ablation run plan: {paths.plan_json}")


@app.command()
def eval_checkpoint_modal(
    run_name: str,
    config_name: str,
    model_type: str = "fusion",
    perturbation: str = "none",
    seed: int = 0,
    missing_rate: float = 0.5,
    eval_run_name: str | None = None,
    eval_group: str = "ablations/eval",
) -> None:
    import subprocess

    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "-m",
        "src.use_modal::evaluate_checkpoint",
        f"--run-name={run_name}",
        f"--config-name={config_name}",
        f"--model-type={model_type}",
        f"--perturbation={perturbation}",
        f"--seed={seed}",
        f"--missing-rate={missing_rate}",
        f"--eval-group={eval_group}",
    ]
    if eval_run_name is not None:
        cmd.append(f"--eval-run-name={eval_run_name}")
    subprocess.run(cmd, check=True)


@app.command()
def explainability_modal(
    run_name: str,
    config_name: str,
    model_type: str = "fusion",
    figure_name: str = "figure_explainability_cases.png",
    json_name: str = "explainability_cases.json",
    steps: int = 24,
) -> None:
    import subprocess

    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "-m",
        "src.use_modal::generate_explainability_cases",
        f"--run-name={run_name}",
        f"--config-name={config_name}",
        f"--model-type={model_type}",
        "--output-dir=/data/buffer",
        f"--figure-name={figure_name}",
        f"--json-name={json_name}",
        f"--steps={steps}",
    ]
    subprocess.run(cmd, check=True)


@app.command()
def phase1() -> None:
    harvest()
    tables()
    figures()
    scaffold_ablation()
    ablation_plan()


if __name__ == "__main__":
    app()
