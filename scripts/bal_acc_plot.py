import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer

from src.pipelines.label_config import LABEL_CONFIG, SIMILAR_PAIRS

app = typer.Typer()

NAME_TO_SHORT: dict[str, str] = {v["name"]: v["short"] for v in LABEL_CONFIG.values()}
SHORT_TO_NAME: dict[str, str] = {v["short"]: v["name"] for v in LABEL_CONFIG.values()}

FIGURES_DIR = Path(__file__).parent.parent / "assets" / "figures"


def build_confusion_matrix(df: pl.DataFrame, classes: list[str]) -> np.ndarray:
    """Build a confusion matrix from a DataFrame of predictions.

    Args:
        df: DataFrame with columns 'Actual', 'Predicted', and 'nPredictions'.
        classes: List of class names in sorted order.

    Returns:
        Confusion matrix as a 2D numpy array of shape (n_classes, n_classes).
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for row in df.iter_rows(named=True):
        i = classes.index(row["Actual"])
        j = classes.index(row["Predicted"])
        cm[i, j] = row["nPredictions"]
    return cm


def compute_pairwise_accuracy(
    cm: np.ndarray, classes: list[str], class_a: str, class_b: str
) -> float:
    """Compute accuracy on the subset of samples from two classes.

    Calculates: (correct_A + correct_B) / (total_A + total_B)
    This shows what % of samples from these two classes were classified correctly.

    Args:
        cm: Confusion matrix as a 2D numpy array.
        classes: List of class names corresponding to matrix indices.
        class_a: Name of the first class in the pair.
        class_b: Name of the second class in the pair.

    Returns:
        Subset accuracy as a percentage (0-100).
    """
    idx_a = classes.index(class_a)
    idx_b = classes.index(class_b)

    correct_a = cm[idx_a, idx_a]
    correct_b = cm[idx_b, idx_b]
    total_a = cm[idx_a, :].sum()
    total_b = cm[idx_b, :].sum()

    total = total_a + total_b
    if total == 0:
        return 0.0

    return (correct_a + correct_b) / total * 100


BAR_ORDER = ["blue", "red", "purple"]
COLOR_MAP = {
    "red": "#f0434f",
    "blue": "#538ae5",
    "purple": "#865ed6",
}
MAX_PER_ROW = 3


def generate_bar_chart(
    csv_paths: dict[str, Path | None],
    csv_names: dict[str, str],
    output_path: Path,
) -> None:
    """Generate a grouped bar chart of balanced accuracy for visually similar pairs.

    Args:
        csv_paths: Mapping of color keys ('red', 'blue', 'purple') to CSV file paths.
        csv_names: Mapping of color keys to display names for the legend.
        output_path: Path where the output image will be saved.
    """
    data: dict[str, tuple[np.ndarray, list[str]]] = {}
    classes: list[str] | None = None

    for color, path in csv_paths.items():
        if path is not None:
            df = pl.read_csv(path)
            cls = sorted(df["Actual"].unique().to_list())
            if classes is None:
                classes = cls
            cm = build_confusion_matrix(df, cls)
            data[color] = (cm, cls)

    if not data or classes is None:
        typer.echo("Error: At least one CSV file must be provided.")
        raise typer.Exit(1)

    pair_labels: list[str] = []
    pair_scores: dict[str, list[float]] = {color: [] for color in data}

    for short_a, short_b in SIMILAR_PAIRS:
        name_a = SHORT_TO_NAME.get(short_a)
        name_b = SHORT_TO_NAME.get(short_b)
        if name_a in classes and name_b in classes:
            pair_labels.append(f"{short_a} vs {short_b}")
            for color, (cm, cls) in data.items():
                ba = compute_pairwise_accuracy(cm, cls, name_a, name_b)
                pair_scores[color].append(ba)

    if not pair_labels:
        typer.echo("Error: No similar pairs found in the data.")
        raise typer.Exit(1)

    n_pairs = len(pair_labels)
    n_rows = 2
    pairs_per_row = min(MAX_PER_ROW, (n_pairs + n_rows - 1) // n_rows)
    row_splits = [
        pair_labels[:pairs_per_row],
        pair_labels[pairs_per_row:],
    ]
    score_splits = [
        {c: scores[:pairs_per_row] for c, scores in pair_scores.items()},
        {c: scores[pairs_per_row:] for c, scores in pair_scores.items()},
    ]

    row_splits = [r for r in row_splits if r]
    score_splits = [s for s in score_splits if any(s.values())]
    n_rows = len(row_splits)

    ordered_colors = [c for c in BAR_ORDER if c in data]
    n_models = len(ordered_colors)
    width = 0.8 / n_models

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row_idx, (labels, scores_dict) in enumerate(zip(row_splits, score_splits)):
        ax = axes[row_idx]
        n_in_row = len(labels)
        x = np.arange(n_in_row)

        for i, color in enumerate(ordered_colors):
            offset = (i - (n_models - 1) / 2) * width
            row_scores = scores_dict[color]
            bars = ax.bar(
                x + offset,
                row_scores,
                width,
                label=csv_names[color],
                color=COLOR_MAP[color],
                edgecolor="white",
                linewidth=0.5,
            )

            for bar, score in zip(bars, row_scores):
                height = bar.get_height()
                ax.annotate(
                    f"{score:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=n_models,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Accuracy on Visually Similar Class Pairs",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Saved to {output_path}")


@app.command()
def main(
    red_csv: str = typer.Option(
        None,
        help="Path to red model results CSV",
    ),
    blue_csv: str = typer.Option(
        None,
        help="Path to blue model results CSV",
    ),
    purple_csv: str = typer.Option(
        None,
        help="Path to purple model results CSV",
    ),
    red_name: str = typer.Option(
        "Red Model",
        help="Display name for red model",
    ),
    blue_name: str = typer.Option(
        "Blue Model",
        help="Display name for blue model",
    ),
    purple_name: str = typer.Option(
        "Purple Model",
        help="Display name for purple model",
    ),
    output: Path = typer.Option(
        FIGURES_DIR / "balanced_accuracy_bars.png",
        help="Output image path",
    ),
) -> None:
    """Generate bar chart of balanced accuracy on visually similar pairs."""
    csv_paths: dict[str, Path | None] = {
        "red": Path(red_csv.strip("'\"")) if red_csv else None,
        "blue": Path(blue_csv.strip("'\"")) if blue_csv else None,
        "purple": Path(purple_csv.strip("'\"")) if purple_csv else None,
    }
    csv_names: dict[str, str] = {
        "red": red_name,
        "blue": blue_name,
        "purple": purple_name,
    }

    csv_paths = {k: v for k, v in csv_paths.items() if v is not None}
    csv_names = {k: v for k, v in csv_names.items() if k in csv_paths}

    if not csv_paths:
        typer.echo("Error: At least one CSV file must be provided.")
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    generate_bar_chart(csv_paths, csv_names, output)


if __name__ == "__main__":
    app()
