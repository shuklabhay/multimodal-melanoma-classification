import asyncio
import random
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt

from src.pipelines.label_config import ISIC19_LABEL_CONFIG, SPLIT_COLORS
from src.utils.llm_utils import process_narrative_entries

MEMORY_QUOTAS = {
    "vague": 0.25,
    "approximate": 0.30,
    "precise": 0.25,
    "detailed": 0.20,
}

ASYMPTOMATIC_PROB = 0.20
SYMPTOM_COUNT_DIST = {1: 0.40, 2: 0.35, 3: 0.25}


def load_text_file(filepath: str | Path) -> list[str]:
    """Loads lines from a text file, strips whitespace, and removes empty lines.

    Args:
        filepath: Path to the text file to load.

    Returns:
        List of non-empty lines from the file.
    """
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def load_narrative_options(options_dir: Path) -> dict:
    """Loads all narrative option files from the options directory.

    Args:
        options_dir: Path to the directory containing narrative option files.

    Returns:
        Dictionary with loaded narrative options.
    """
    return {
        "styles": load_text_file(options_dir / "style.txt"),
        "memory": load_text_file(options_dir / "memory.txt"),
        "temporal_quantifiers": load_text_file(
            options_dir / "temporal_quantifiers.txt"
        ),
        "structures": load_text_file(options_dir / "structure.txt"),
        "prompt_template": (options_dir / "prompt.txt").read_text(),
    }


def categorize_memory(memory: str) -> str:
    """Categorizes a memory string into higher-level groups for quota tracking.

    Args:
        memory: Memory vividness description to categorize.

    Returns:
        Category name (vague, approximate, precise, or detailed).
    """
    if (
        "vague" in memory.lower()
        or "fragmentary" in memory.lower()
        or "contradictory" in memory.lower()
    ):
        return "vague"
    elif "approximate" in memory.lower() or "anchored" in memory.lower():
        return "approximate"
    elif "precise" in memory.lower() or "exact" in memory.lower():
        return "precise"
    else:
        return "detailed"


def select_narrative_parameters(
    label: int,
    narrative_options: dict,
    rng: random.Random,
) -> dict:
    """Selects narrative parameters for generation using deterministic random selection.

    Args:
        label: Image label (0-7).
        narrative_options: Dictionary with all narrative options.
        rng: Random number generator for deterministic selection.

    Returns:
        Dictionary with selected parameters.
    """
    symptoms_file = ISIC19_LABEL_CONFIG[label]["symptoms"]
    available_symptoms = load_text_file(symptoms_file)

    is_asymptomatic = rng.random() < ASYMPTOMATIC_PROB
    if is_asymptomatic:
        selected_symptoms = []
    else:
        num_symptoms = rng.choices(
            list(SYMPTOM_COUNT_DIST.keys()),
            weights=list(SYMPTOM_COUNT_DIST.values()),
            k=1,
        )[0]
        selected_symptoms = rng.sample(
            available_symptoms, min(num_symptoms, len(available_symptoms))
        )

    style = rng.choice(narrative_options["styles"])
    memory = rng.choice(narrative_options["memory"])
    temporal_quantifier = rng.choice(narrative_options["temporal_quantifiers"])
    structure = rng.choice(narrative_options["structures"])

    return {
        "style": style,
        "memory": memory,
        "temporal_quantifier": temporal_quantifier,
        "symptoms": selected_symptoms,
        "memory_category": categorize_memory(memory),
        "structure": structure,
    }


def create_local_narrative_manifests() -> None:
    """Generates complete narrative manifests with all parameters locally (no text generation).

    Creates narrative manifests with all metadata and parameters pre-selected for each image.
    Text content is left empty to be filled by Modal generation later.
    """
    options_dir = Path("assets/narrative_options")
    narrative_options = load_narrative_options(options_dir)

    rng = random.Random(42)
    manifests_dir = Path("assets/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        image_manifest_path = manifests_dir / f"isic_split_{split}.jsonl"
        if not image_manifest_path.exists():
            continue

        with jsonlines.open(image_manifest_path) as reader:
            image_entries = list(reader)

        narrative_manifest_path = manifests_dir / f"narrative_split_{split}.jsonl"
        narrative_entries = []

        for entry in image_entries:
            label = entry["label"]
            image_path = Path(entry["image_path"])

            text_filename = image_path.stem + ".txt"
            text_path_train = f"/data/raw/train/narratives/{text_filename}"
            text_path_test = f"/data/raw/test/narratives/{text_filename}"
            text_path = text_path_train if split == "train" else text_path_test

            params = select_narrative_parameters(label, narrative_options, rng)
            narrative_entry = {
                "text_path": text_path,
                "text": "",
                "label": label,
                "patient_id": entry.get("patient_id", ""),
                "style": params["style"],
                "memory": params["memory"],
                "memory_category": params["memory_category"],
                "temporal_quantifier": params["temporal_quantifier"],
                "symptoms": params["symptoms"],
                "structure": params["structure"],
                "age_approx": entry.get("age_approx"),
                "sex": entry.get("sex"),
                "anatom_site_general": entry.get("anatom_site_general"),
            }

            narrative_entries.append(narrative_entry)

        with jsonlines.open(narrative_manifest_path, mode="w") as writer:
            writer.write_all(narrative_entries)


def visualize_narrative_manifests() -> None:
    """Generates and saves narrative distribution visualizations from local manifests."""
    manifests_dir = Path("assets/manifests")
    figures_dir = Path("assets/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_manifests = {}
    splits = ["train", "val", "test"]

    for split in splits:
        manifest_path = manifests_dir / f"narrative_split_{split}.jsonl"
        if manifest_path.exists():
            with jsonlines.open(manifest_path) as reader:
                all_manifests[split] = list(reader)

    if all_manifests:
        _generate_distribution_summaries(all_manifests, figures_dir)


def generate_narrative_texts() -> None:
    """Generates narrative text content from pre-generated manifests on Modal."""
    options_dir = Path("/root/assets/narrative_options")
    prompt_template = (options_dir / "prompt.txt").read_text()

    base_dir = Path("/data/raw")
    manifests_dir = Path("/root/assets/manifests")

    splits = ["train", "val", "test"]
    for split in splits:
        print(f"Starting generation for {split} split...")
        narratives_dir = (
            base_dir / ("train" if split == "train" else "test") / "narratives"
        )
        narratives_dir.mkdir(parents=True, exist_ok=True)

        narrative_manifest_path = manifests_dir / f"narrative_split_{split}.jsonl"
        if not narrative_manifest_path.exists():
            continue

        with jsonlines.open(narrative_manifest_path) as reader:
            narrative_entries = list(reader)

        total_entries = asyncio.run(
            process_narrative_entries(
                narrative_entries, prompt_template, split, batch_size=4
            )
        )
        print(f"  Completed {split} split: {total_entries} narratives generated")

    print("\n✓ Narrative text generation complete!")


def verify_narrative_manifests() -> None:
    """Verifies narrative manifests for data quality and quota adherence."""
    print("\nVerifying narrative manifests...")
    manifests_dir = Path("assets/manifests")

    if not manifests_dir.exists():
        print("  ⚠ Manifests directory not found. Run generation first.")
        return

    manifest_files = list(manifests_dir.glob("narrative_split_*.jsonl"))

    if not manifest_files:
        print("  ⚠ No narrative manifest files found.")
        return

    all_manifests = {}
    for manifest_file in sorted(manifest_files):
        split_name = manifest_file.stem.replace("narrative_split_", "")
        print(f"\n  Verifying {split_name} manifest...")

        entries = []
        with jsonlines.open(manifest_file) as reader:
            entries = list(reader)

        print(f"    Total entries: {len(entries)}")

        missing_texts = sum(1 for e in entries if not Path(e["text_path"]).exists())
        if missing_texts > 0:
            print(f"    ⚠ WARNING: {missing_texts} text files not found")
        else:
            print(f"    ✓ All text files exist")

        symptom_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for entry in entries:
            num_symptoms = len(entry.get("symptoms", []))

            symptom_counts[num_symptoms] = symptom_counts.get(num_symptoms, 0) + 1

        print(f"    Symptom count distribution:")
        for num, count in sorted(symptom_counts.items()):
            pct = (count / len(entries)) * 100
            print(f"      {num} symptoms: {count} ({pct:.1f}%)")

        all_manifests[split_name] = entries

    print("\n✓ Verification complete")

    print("\nGenerating distribution visualizations...")
    figures_dir = Path("assets/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _generate_distribution_summaries(all_manifests, figures_dir)
    print(f"Visualizations saved to {figures_dir}")


def _generate_distribution_summaries(
    manifests: dict[str, list[dict]], output_dir: Path
) -> None:
    """Generates and saves narrative distribution visualizations.

    Args:
        manifests: Dictionary with split names as keys and manifest entries as values.
        output_dir: Directory to save visualization images.
    """
    label_names = {
        label_id: config["name"] for label_id, config in ISIC19_LABEL_CONFIG.items()
    }

    _plot_symptom_distribution(manifests, label_names, output_dir)


def _plot_symptom_distribution(
    manifests: dict[str, list[dict]], label_names: dict, output_dir: Path
) -> None:
    """Plots symptom count distribution across splits."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    symptom_counts_range = [0, 1, 2, 3]
    x = range(len(symptom_counts_range))
    width = 0.25

    for i, split_name in enumerate(["train", "val", "test"]):
        if split_name not in manifests:
            continue

        entries = manifests[split_name]
        total_samples = len(entries)

        symptom_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for entry in entries:
            num_symptoms = len(entry.get("symptoms", []))
            symptom_counts[num_symptoms] = symptom_counts.get(num_symptoms, 0) + 1

        counts = [symptom_counts.get(num, 0) for num in symptom_counts_range]
        percentages = [(count / total_samples) * 100 for count in counts]

        offset = width * (i - 1)
        ax.bar(
            [xi + offset for xi in x],
            percentages,
            width,
            label=f"{split_name.capitalize()} (n={total_samples})",
            color=SPLIT_COLORS[split_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(num) for num in symptom_counts_range])
    ax.set_xlabel("Number of Symptoms")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Symptom Count Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "balance_symptoms.png", dpi=150)
    plt.close()
