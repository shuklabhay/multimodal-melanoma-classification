import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import yaml
from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()

ModelType = Literal["image", "fusion", "text"]
RUN_MODE_TO_MODEL_TYPE = {
    "image_only": "image",
    "multimodal": "fusion",
    "text_only": "text",
}


def compute_class_weights(
    manifest_path: str = "assets/manifests/split_train.jsonl",
    output_path: str = "assets/class_weights.npy",
) -> None:
    """Compute class weights from training manifest and save to file.

    Args:
        manifest_path: Path to training manifest JSONL.
        output_path: Path to save class weights as .npy file.
    """
    labels = []
    with open(manifest_path) as f:
        for line in f:
            item = json.loads(line)
            labels.append(item["label"])

    counts = Counter(labels)
    num_classes = len(counts)
    total = sum(counts.values())

    print("\n=== Class Distribution ===")
    for c in range(num_classes):
        freq = counts[c] / total
        print(f"  Class {c}: {counts[c]:>5} samples ({freq:.1%})")

    weights = np.array([total / (num_classes * counts[c]) for c in range(num_classes)])
    weights = weights / weights.sum() * num_classes

    print("\n=== Class Weights ===")
    for c in range(num_classes):
        print(f"  Class {c}: {weights[c]:.4f}")

    np.save(output_path, weights)
    print(f"\n✓ Saved class weights to {output_path}")


def list_configs() -> list[Path]:
    """List all YAML config files in the config directory."""
    return sorted(Path("config").glob("*.yaml"))


def select_config() -> Path:
    """Prompt user to select a config file."""
    configs = list_configs()
    if not configs:
        raise FileNotFoundError("No config files found in config/")

    print("\n=== Available Configs ===")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config.stem}")

    while True:
        choice = input("\nSelect config number: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                return configs[idx]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Try again.")


def load_raw_config(config_path: Path) -> dict:
    """Load raw YAML config for CLI decisions."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def resolve_config_model_type(config: dict) -> ModelType:
    """Resolve model type from config (`type` preferred, legacy `run_mode` supported)."""
    model_type = config.get("type")
    if model_type in {"image", "fusion", "text"}:
        return cast(ModelType, model_type)

    legacy_mode = config.get("run_mode")
    mapped = RUN_MODE_TO_MODEL_TYPE.get(legacy_mode)
    if mapped is not None:
        return cast(ModelType, mapped)

    raise ValueError(
        "Config must contain `type` (image|fusion|text) or "
        "legacy `run_mode` (image_only|multimodal|text_only)"
    )


def select_model_type(config_path: Path) -> ModelType:
    """Prompt user to select model type for the selected config."""
    config = load_raw_config(config_path)
    default_type = resolve_config_model_type(config)

    has_image = config.get("image_enc") is not None
    has_text = config.get("text_enc") is not None

    options: list[tuple[str, ModelType, str]] = []
    if has_image and has_text:
        options.append(("Image only", "image", "Use only image backbone features"))
        options.append(("Fusion", "fusion", "Use image + text (concat features)"))
    elif has_image:
        options.append(("Image only", "image", "Use only image backbone features"))
    elif has_text:
        options.append(("Text only", "text", "Use only text embeddings"))

    if not options:
        raise ValueError(
            "Selected config has neither image_enc nor text_enc. Cannot select model type."
        )

    supported_types = {opt[1] for opt in options}
    if default_type not in supported_types:
        default_type = options[0][1]

    default_idx = next(
        i
        for i, (_, opt_type, _) in enumerate(options, start=1)
        if opt_type == default_type
    )

    print("\n=== Training Type ===")
    for i, (label, opt_type, description) in enumerate(options, start=1):
        default_marker = " (default)" if opt_type == default_type else ""
        print(f"  {i}. {label}{default_marker} - {description}")

    while True:
        choice = input(f"\nSelect type number [{default_idx}]: ").strip()
        if choice == "":
            return default_type
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Try again.")


@app.command()
def train() -> None:
    """Launch training on Modal with selected config and model type."""
    config_path = select_config()
    model_type = select_model_type(config_path)
    timestamp = datetime.now().strftime("%b%d%H%M").lower()
    run_name = f"{config_path.stem}_{model_type}_{timestamp}"

    print(f"\nStarting training with config: {config_path.stem}")
    print(f"Model type: {model_type}")
    print(f"Run name: {run_name}")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "--detach",
            "-m",
            "src.use_modal::train_model",
            f"--run-name={run_name}",
            f"--config-name={config_path.stem}",
            f"--model-type={model_type}",
        ],
        check=True,
    )


def fetch_checkpoints() -> list[dict]:
    """Fetch available checkpoints from Modal volume."""
    print("\nFetching checkpoints from Modal volume...")

    result = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "data", "/checkpoints", "--json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return []

    import json

    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    checkpoints = []
    for entry in entries:
        if entry.get("Type") == "dir":
            filename = entry.get("Filename", "")
            run_name = filename.split("/")[-1] if filename else ""
            if run_name:
                checkpoints.append({"run_name": run_name})

    return checkpoints


def select_checkpoint(checkpoints: list[dict]) -> dict | None:
    """Prompt user to select a checkpoint to resume from."""
    if not checkpoints:
        print("\nNo checkpoints available.")
        return None

    print("\n=== Available Checkpoints ===")
    for i, ckpt in enumerate(checkpoints, 1):
        run_name = ckpt.get("run_name", "unknown")
        print(f"  {i}. {run_name}")

    print(f"  {len(checkpoints) + 1}. Cancel")

    while True:
        choice = input("\nSelect checkpoint number: ").strip()
        try:
            idx = int(choice) - 1
            if idx == len(checkpoints):
                return None
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Try again.")


def resume() -> None:
    """Resumes training from a selected checkpoint by prompting user and launching Modal."""
    checkpoints = fetch_checkpoints()
    checkpoint = select_checkpoint(checkpoints)

    if checkpoint is None:
        print("Resume cancelled.")
        return

    resume_run_name = checkpoint["run_name"]

    config_dir = Path("config")
    config_name = None
    for config_file in config_dir.glob("*.yaml"):
        potential_config = config_file.stem
        if resume_run_name.startswith(potential_config + "_"):
            config_name = potential_config
            break

    if not config_name:
        print(f"Error: Could not determine config for checkpoint {resume_run_name}.")
        return

    print(f"\nResuming: {resume_run_name}")
    print(f"Config: {config_name}")

    resume_model_type = None
    run_name_parts = resume_run_name.split("_")
    if len(run_name_parts) >= 3 and run_name_parts[-2] in {"image", "fusion", "text"}:
        resume_model_type = run_name_parts[-2]
        print(f"Model type: {resume_model_type}")

    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "--detach",
        "-m",
        "src.use_modal::train_model",
        f"--run-name={resume_run_name}",
        f"--config-name={config_name}",
        f"--resume-from={resume_run_name}",
    ]
    if resume_model_type is not None:
        cmd.append(f"--model-type={resume_model_type}")

    subprocess.run(cmd, check=True)


def bundle() -> None:
    """Bundles data splits into NPZ files by launching bundling job on Modal."""
    print("\nStarting data bundling on Modal...")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::bundle_splits",
        ],
        check=True,
    )


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for training utilities."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== Training Utilities ===")
    print("  1. Bundle data")
    print("  2. Compute class weights")
    print("  3. Train model")
    print("  4. Resume training")
    print("  5. Exit")

    steps_map = {
        "1": bundle,
        "2": compute_class_weights,
        "3": train,
        "4": resume,
    }

    choice = input("\nEnter your choice: ").strip()

    if choice == "5":
        print("Exiting...")
        return

    if choice in steps_map:
        steps_map[choice]()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
