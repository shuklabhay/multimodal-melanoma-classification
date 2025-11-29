import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Typer()


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for training utilities."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== Training Utilities ===")
    print("No commands available.")
    print("  1. Exit")
    choice = input("\nEnter your choice: ").strip().lower()

    if choice == "1":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
