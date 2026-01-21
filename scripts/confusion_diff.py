import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src.pipelines.dataset_config import ISIC19_LABEL_CONFIG, ISIC19_SIMILAR_PAIRS

app = typer.Typer()


NAME_TO_SHORT: dict[str, str] = {
    v["name"]: v["short"] for v in ISIC19_LABEL_CONFIG.values()
}
VISUALLY_SIMILAR_SHORTS: set[str] = set()
for a, b in ISIC19_SIMILAR_PAIRS:
    VISUALLY_SIMILAR_SHORTS.add(a)
    VISUALLY_SIMILAR_SHORTS.add(b)

FIGURES_DIR = Path(__file__).parent.parent / "assets" / "figures"


def build_confusion_matrix(df: pl.DataFrame, classes: list[str]) -> np.ndarray:
    """Builds a confusion matrix from prediction DataFrame.

    Args:
        df: DataFrame with 'Actual', 'Predicted', and 'nPredictions' columns.
        classes: List of class names in sorted order.

    Returns:
        Confusion matrix as 2D numpy array of shape (n_classes, n_classes).
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for row in df.iter_rows(named=True):
        i = classes.index(row["Actual"])
        j = classes.index(row["Predicted"])
        cm[i, j] = row["nPredictions"]
    return cm


def fdr_correction(pvals: np.ndarray) -> np.ndarray:
    """Applies Benjamini-Hochberg FDR correction to p-values.

    Args:
        pvals: Array of raw p-values.

    Returns:
        Array of FDR-corrected p-values with same shape as input.
    """
    pvals = np.array(pvals).flatten()
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    corrected = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        cummin = min(cummin, sorted_pvals[i] * n / (i + 1))
        corrected[sorted_idx[i]] = min(cummin, 1.0)
    return corrected


def compute_pvalues(cm_a: np.ndarray, cm_b: np.ndarray) -> np.ndarray:
    """Computes p-values for pairwise proportion comparison between confusion matrices.

    Args:
        cm_a: Confusion matrix for model A.
        cm_b: Confusion matrix for model B.

    Returns:
        Matrix of p-values of shape (n_classes, n_classes).
    """
    n_classes = cm_a.shape[0]
    pvals = np.ones((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            n1, n2 = cm_a[i, j], cm_b[i, j]
            total1, total2 = cm_a[i, :].sum(), cm_b[i, :].sum()
            if total1 > 0 and total2 > 0:
                p1, p2 = n1 / total1, n2 / total2
                p_pool = (n1 + n2) / (total1 + total2)
                if 0 < p_pool < 1:
                    se = np.sqrt(p_pool * (1 - p_pool) * (1 / total1 + 1 / total2))
                    if se > 0:
                        z = (p1 - p2) / se
                        pvals[i, j] = 2 * (1 - stats.norm.cdf(abs(z)))
    return pvals


def generate_heatmap(
    model_a_csv: Path,
    model_b_csv: Path,
    model_a_name: str,
    model_b_name: str,
    output_path: Path,
) -> None:
    """Generates confusion difference heatmap comparing two models.

    Args:
        model_a_csv: Path to Model A results CSV.
        model_b_csv: Path to Model B results CSV.
        model_a_name: Display name for Model A.
        model_b_name: Display name for Model B.
        output_path: Path to save the output image.
    """
    model_a = pl.read_csv(model_a_csv)
    model_b = pl.read_csv(model_b_csv)

    classes: list[str] = sorted(model_a["Actual"].unique().to_list())
    n_classes = len(classes)
    labels = [NAME_TO_SHORT.get(c, c[:4]) for c in classes]

    cm_a = build_confusion_matrix(model_a, classes)
    cm_b = build_confusion_matrix(model_b, classes)

    row_totals_a = cm_a.sum(axis=1, keepdims=True)
    row_totals_b = cm_b.sum(axis=1, keepdims=True)
    cm_a_pct = np.where(row_totals_a > 0, cm_a / row_totals_a * 100, 0.0)
    cm_b_pct = np.where(row_totals_b > 0, cm_b / row_totals_b * 100, 0.0)
    diff = cm_a_pct - cm_b_pct

    pvals = compute_pvalues(cm_a, cm_b)
    pvals_corrected = fdr_correction(pvals).reshape((n_classes, n_classes))

    similar_class_indices: set[int] = set()
    for i, c in enumerate(classes):
        short = NAME_TO_SHORT.get(c, "")
        if short in VISUALLY_SIMILAR_SHORTS:
            similar_class_indices.add(i)

    diagonal_increase = [(i, i) for i in similar_class_indices]
    off_diagonal_decrease: list[tuple[int, int]] = []
    short_to_idx = {NAME_TO_SHORT.get(c, ""): i for i, c in enumerate(classes)}
    for a, b in ISIC19_SIMILAR_PAIRS:
        if a in short_to_idx and b in short_to_idx:
            idx_a, idx_b = short_to_idx[a], short_to_idx[b]
            off_diagonal_decrease.append((idx_a, idx_b))
            off_diagonal_decrease.append((idx_b, idx_a))

    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = max(abs(diff.max()), abs(diff.min()))
    rdylgn = LinearSegmentedColormap.from_list(
        "RdYlGn",
        ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
    )
    im = ax.imshow(diff, cmap=rdylgn, vmin=-vmax, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{model_a_name} - {model_b_name} (%)", fontsize=11)

    for i in range(n_classes):
        for j in range(n_classes):
            val = diff[i, j]
            pval = pvals_corrected[i, j]
            sig = ""
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"

            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:+.1f}%{sig}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    inset = 0.01
    for i, j in diagonal_increase:
        rect = patches.Rectangle(
            (j - 0.5 + inset, i - 0.5 + inset),
            1 - 2 * inset,
            1 - 2 * inset,
            linewidth=2,
            edgecolor="#1a9850",
            facecolor="none",
        )
        ax.add_patch(rect)

    for i, j in off_diagonal_decrease:
        rect = patches.Rectangle(
            (j - 0.5 + inset, i - 0.5 + inset),
            1 - 2 * inset,
            1 - 2 * inset,
            linewidth=2,
            edgecolor="#d73027",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Difference: {model_a_name} - {model_b_name}\n"
        "Green outline = want ↑ | Red outline = want ↓ | *p<.05 **p<.01 ***p<.001",
        fontsize=11,
        pad=40,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Saved to {output_path}")


@app.command()
def main(
    model_a_csv: str = typer.Option(
        ...,
        prompt="Path to Model A (e.g. Fusion) results CSV",
        help="Path to Model A results CSV",
    ),
    model_b_csv: str = typer.Option(
        ...,
        prompt="Path to Model B (e.g. Baseline) results CSV",
        help="Path to Model B results CSV",
    ),
    model_a_name: str = typer.Option(
        ..., prompt="Name for Model A (e.g. Fusion)", help="Display name for Model A"
    ),
    model_b_name: str = typer.Option(
        ...,
        prompt="Name for Model B (e.g. Image-Only)",
        help="Display name for Model B",
    ),
    output: Path = typer.Option(
        FIGURES_DIR / "confusion_difference_heatmap.png",
        help="Output image path",
    ),
) -> None:
    """Generate confusion difference heatmap (Model A - Model B)."""
    path_a = Path(model_a_csv.strip("'\""))
    path_b = Path(model_b_csv.strip("'\""))
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_heatmap(path_a, path_b, model_a_name, model_b_name, output)


if __name__ == "__main__":
    app()
