import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_utils import use_buffer

app = Typer()


@app.command()
def test_tsne() -> None:
    """Generate t-SNE visualization of ResNet50 embeddings for test set."""
    print("\nGenerating test set t-SNE on Modal...")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::generate_test_tsne",
        ],
        check=True,
    )

    use_buffer(
        {
            "test_tsne.png": "assets/figures/test_tsne.png",
        }
    )
    print("\nt-SNE visualization saved to assets/figures/test_tsne.png")


@app.command()
def similar_pairs() -> None:
    """Generate k-NN mixing analysis and similar pairs visualization."""
    print("\nGenerating similar pairs analysis on Modal...")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::generate_similar_pairs",
        ],
        check=True,
    )

    use_buffer(
        {
            "test_similar_pairs.png": "assets/figures/test_similar_pairs.png",
        }
    )
    print("\nSimilar pairs saved to assets/figures/test_similar_pairs.png")


@app.command()
def class_separation() -> None:
    """Generate class separation analysis (silhouette, Davies-Bouldin, distance distributions)."""
    print("\nGenerating class separation analysis on Modal...")
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::generate_class_separation_analysis",
        ],
        check=True,
    )

    use_buffer(
        {
            "class_separation_analysis.png": "assets/figures/class_separation_analysis.png",
            "class_separation_metrics.json": "assets/class_separation_metrics.json",
        }
    )
    print("\nClass separation analysis saved to assets/figures/class_separation_analysis.png")
    print("Metrics saved to assets/class_separation_metrics.json")


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for analysis utilities."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== Analysis Utilities ===")
    print("What would you like to do?")
    print("  1. Generate test set t-SNE")
    print("  2. Generate similar pairs visual")
    print("  3. Generate class separation analysis")
    print("  4. Exit")

    steps_map = {
        "1": test_tsne,
        "2": similar_pairs,
        "3": class_separation,
    }

    choice = input("\nEnter your choice: ").strip()

    if choice == "4":
        print("Exiting...")
        return

    if choice in steps_map:
        steps_map[choice]()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    app()
