import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()


@app.command("verify-dataloaders")
def verify_dataloaders(
    num_batches: int = 2,
    batch_size: int = 4,
) -> None:
    """Launches interactive visualizer to inspect dataloader outputs.

    Args:
        num_batches: Number of batches to fetch per split.
        batch_size: Batch size for dataloaders.
    """
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::visualizer_entrypoint",
            "--num-batches",
            str(num_batches),
            "--batch-size",
            str(batch_size),
        ],
        check=True,
    )


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for training utilities."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== Training Utilities ===")
    print("What would you like to do?")
    print("  1. Verify dataloaders (visualizer)")
    print("  2. Exit")
    choice = input("\nEnter your choice: ").strip().lower()

    steps_map = {
        "1": verify_dataloaders,
    }

    if choice == "2":
        print("Exiting...")
        return

    if "," in choice:
        selected_steps = [s.strip() for s in choice.split(",")]
        invalid_steps = [s for s in selected_steps if s not in steps_map]

        if invalid_steps:
            print(f"Invalid step(s): {', '.join(invalid_steps)}. Please use 1-2.")
            return

        print(f"\nRunning steps: {', '.join(selected_steps)}\n")
        for step_num in selected_steps:
            print(f"Running step {step_num}...")
            steps_map[step_num]()
    elif choice in steps_map:
        steps_map[choice]()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
