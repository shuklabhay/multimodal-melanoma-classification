import subprocess
import sys
from datetime import datetime
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()


def run_eval(modality: str) -> None:
    """Run GPT-5 evaluation for a specific modality.

    Args:
        modality: "image", "text", or "fusion".
    """
    timestamp = datetime.now().strftime("%b%d%H%M").lower()
    run_name = f"{modality}_gpt5_chat_{timestamp}"

    print(f"\nStarting GPT-5 evaluation")
    print(f"  Modality: {modality}")
    print(f"  Run name: {run_name}")

    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "--detach",
        "-m",
        "src.use_modal::evaluate_gpt5_on_test",
        f"--modality={modality}",
        f"--run-name={run_name}",
    ]

    subprocess.run(cmd, check=True)


@app.command()
def image() -> None:
    """Evaluate GPT-5 on image-only inputs."""
    run_eval("image")


@app.command()
def text() -> None:
    """Evaluate GPT-5 on text-only inputs (patient narratives)."""
    run_eval("text")


@app.command()
def fusion() -> None:
    """Evaluate GPT-5 on image+text fusion inputs."""
    run_eval("fusion")


@app.command()
def all() -> None:
    """Run GPT-5 evaluation for all modalities sequentially."""
    for modality in ["image", "text", "fusion"]:
        run_eval(modality)


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for GPT-5 evaluation."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== GPT-5 Evaluation ===")
    print("  1. Image only")
    print("  2. Text only")
    print("  3. Fusion (image + text)")
    print("  4. All modalities")
    print("  5. Exit")

    choice = input("\nSelect evaluation type: ").strip()

    if choice == "5":
        print("Exiting...")
        return

    modality_map = {
        "1": "image",
        "2": "text",
        "3": "fusion",
    }

    if choice == "4":
        for modality in ["image", "text", "fusion"]:
            run_eval(modality)
    elif choice in modality_map:
        run_eval(modality_map[choice])
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
