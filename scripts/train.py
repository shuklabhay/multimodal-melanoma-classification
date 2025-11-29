import subprocess
import sys
from datetime import datetime
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()


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


@app.command()
def train() -> None:
    """Launch training on Modal with selected config."""
    config_path = select_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{config_path.stem}"

    print(f"\nStarting training with config: {config_path.stem}")
    print(f"Run name: {run_name}")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::train_model",
            f"--run-name={run_name}",
            f"--config-name={config_path.stem}",
        ],
        check=True,
    )


def bundle() -> None:
    """Bundle data splits into NPZ files on Modal."""
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
    print("  2. Train model")
    print("  3. Exit")
    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        bundle()
    elif choice == "2":
        train()
    elif choice == "3":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
