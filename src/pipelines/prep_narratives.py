import asyncio
import random
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt

from src.pipelines.dataset_config import SPLIT_COLORS, DatasetConfig
from src.utils.llm_utils import process_narrative_entries

MEMORY_QUOTAS = {
    "vague": 0.25,
    "approximate": 0.30,
    "precise": 0.25,
    "detailed": 0.20,
}

ASYMPTOMATIC_PROB = 0.20
SYMPTOM_COUNT_DIST = {1: 0.40, 2: 0.35, 3: 0.25}
ACTION_PROB = 0.65
SYMPTOM_POOL_WEIGHTS = {"universal": 0.65, "label": 0.30, "distractor": 0.05}
HEALTHY_MODE_PROB = 0.15


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
        "actions": load_text_file(options_dir / "action.txt"),
        "symptoms_universal": load_text_file(
            options_dir / "symptoms" / "universal.txt"
        ),
        "symptoms_distractor": load_text_file(
            options_dir / "symptoms" / "distractor.txt"
        ),
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


def _parse_age(age_value: object) -> int | None:
    """Parse a numeric age value into a positive integer when possible.

    Args:
        age_value: Raw age value from metadata.

    Returns:
        Parsed integer age if valid and positive; otherwise None.
    """
    if age_value is None:
        return None
    if isinstance(age_value, (int, float)):
        age_int = int(age_value)
        return age_int if age_int > 0 else None
    if isinstance(age_value, str):
        raw = age_value.strip()
        if not raw:
            return None
        numeric = raw.replace(".", "", 1)
        if numeric.isdigit():
            age_int = int(float(raw))
            return age_int if age_int > 0 else None
    return None


def _build_context_slot(metadata: dict) -> str:
    """Build raw context slot from metadata without templated phrasing.

    Args:
        metadata: Metadata dict with optional age/sex/site fields.

    Returns:
        Context slot string or "none" if no metadata is available.
    """
    age_int = _parse_age(metadata.get("age_approx"))
    sex_value = metadata.get("sex")
    site_value = metadata.get("anatom_site_general")

    parts = []
    if age_int is not None:
        parts.append(f"age_approx={age_int}")
    if isinstance(sex_value, str) and sex_value.strip():
        parts.append(f"sex={sex_value.strip()}")
    if isinstance(site_value, str) and site_value.strip():
        parts.append(f"anatom_site_general={site_value.strip()}")

    return ", ".join(parts) if parts else "none"




def select_narrative_parameters(
    label: int,
    narrative_options: dict,
    rng: random.Random,
    config: DatasetConfig,
    metadata: dict,
) -> dict:
    """Randomly selects narrative generation parameters for a given label.

    Args:
        label: Integer label for the lesion class.
        narrative_options: Dictionary containing available styles, memory types, etc.
        rng: Random number generator for reproducible selection.
        config: Dataset configuration with label-specific symptom files.
        metadata: Patient metadata for context slot construction.

    Returns:
        Dictionary containing selected parameters for narrative generation.
    """
    label_cfg = config.label_config[label]
    narrative_mode = "healthy" if rng.random() < HEALTHY_MODE_PROB else "lesion"
    symptoms_file = label_cfg["symptoms"]
    label_symptoms = load_text_file(symptoms_file)

    if narrative_mode == "healthy":
        selected_symptoms = []
        action_cues = (
            rng.choice(narrative_options["actions"])
            if rng.random() < ACTION_PROB
            else "none"
        )
        structure = rng.choice(narrative_options["structures"])
    else:
        is_asymptomatic = rng.random() < ASYMPTOMATIC_PROB
        if is_asymptomatic:
            selected_symptoms = []
        else:
            num_symptoms = rng.choices(
                list(SYMPTOM_COUNT_DIST.keys()),
                weights=list(SYMPTOM_COUNT_DIST.values()),
                k=1,
            )[0]
            pool_choices = rng.choices(
                list(SYMPTOM_POOL_WEIGHTS.keys()),
                weights=list(SYMPTOM_POOL_WEIGHTS.values()),
                k=num_symptoms,
            )
            if "universal" not in pool_choices and narrative_options[
                "symptoms_universal"
            ]:
                pool_choices[0] = "universal"

            selected_symptoms = []
            for pool_name in pool_choices:
                if pool_name == "universal":
                    pool = narrative_options["symptoms_universal"]
                elif pool_name == "distractor":
                    pool = narrative_options["symptoms_distractor"]
                else:
                    pool = label_symptoms

                if not pool:
                    pool = narrative_options["symptoms_universal"] or label_symptoms
                if pool:
                    selected_symptoms.append(rng.choice(pool))

            selected_symptoms = list(dict.fromkeys(selected_symptoms))

        action_cues = (
            rng.choice(narrative_options["actions"])
            if rng.random() < ACTION_PROB
            else "none"
        )
        structure = rng.choice(narrative_options["structures"])

    style = rng.choice(narrative_options["styles"])
    memory = rng.choice(narrative_options["memory"])
    temporal_quantifier = rng.choice(narrative_options["temporal_quantifiers"])
    context_slot = _build_context_slot(metadata)

    return {
        "style": style,
        "memory": memory,
        "temporal_quantifier": temporal_quantifier,
        "symptoms": selected_symptoms,
        "memory_category": categorize_memory(memory),
        "structure": structure,
        "action_cues": action_cues,
        "context_slot": context_slot,
        "narrative_mode": narrative_mode,
    }


def create_local_narrative_manifests(config: DatasetConfig) -> None:
    """Creates narrative manifest files for each split by assigning generation parameters.

    Args:
        config: Dataset configuration specifying manifest paths and label config.
    """
    options_dir = Path("assets/narrative_options")
    narrative_options = load_narrative_options(options_dir)

    rng = random.Random(42)
    manifests_dir = Path("assets/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    prefix = config.name
    splits = ["train", "val", "test"]
    for split in splits:
        image_manifest_path = manifests_dir / f"{prefix}_split_{split}.jsonl"
        if not image_manifest_path.exists():
            continue

        with jsonlines.open(image_manifest_path) as reader:
            image_entries = list(reader)

        narrative_manifest_path = (
            manifests_dir / f"{prefix}_narratives_split_{split}.jsonl"
        )
        narrative_entries = []

        for entry in image_entries:
            label = entry["label"]
            image_path = Path(entry["image_path"])

            text_filename = image_path.stem + ".txt"
            text_path_train = f"/data/raw/train/{prefix}_narratives/{text_filename}"
            text_path_test = f"/data/raw/test/{prefix}_narratives/{text_filename}"
            text_path = text_path_train if split in ["train", "val"] else text_path_test

            metadata = {
                "age_approx": entry.get("age_approx"),
                "sex": entry.get("sex"),
                "anatom_site_general": entry.get("anatom_site_general"),
            }
            params = select_narrative_parameters(
                label, narrative_options, rng, config, metadata
            )
            narrative_entry = {
                "text_path": text_path,
                "text": "",
                "label": label,
                config.id_column_for_manifest: entry.get(
                    config.id_column_for_manifest, ""
                ),
                "style": params["style"],
                "memory": params["memory"],
                "memory_category": params["memory_category"],
                "temporal_quantifier": params["temporal_quantifier"],
                "symptoms": params["symptoms"],
                "structure": params["structure"],
                "action_cues": params["action_cues"],
                "context_slot": params["context_slot"],
                "narrative_mode": params["narrative_mode"],
                "age_approx": entry.get("age_approx"),
                "sex": entry.get("sex"),
                "anatom_site_general": entry.get("anatom_site_general"),
            }

            narrative_entries.append(narrative_entry)

        with jsonlines.open(narrative_manifest_path, mode="w") as writer:
            writer.write_all(narrative_entries)

    print(f"✓ Created narrative manifests for {prefix}")


def visualize_narrative_manifests(config: DatasetConfig) -> None:
    """Generates distribution visualizations from narrative manifests.

    Args:
        config: Dataset configuration specifying manifest paths and prefix.
    """
    manifests_dir = Path("assets/manifests")
    figures_dir = Path("assets/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    prefix = config.name
    all_manifests = {}
    splits = ["train", "val", "test"]

    for split in splits:
        manifest_path = manifests_dir / f"{prefix}_narratives_split_{split}.jsonl"
        if manifest_path.exists():
            with jsonlines.open(manifest_path) as reader:
                all_manifests[split] = list(reader)

    if all_manifests:
        _generate_distribution_summaries(all_manifests, figures_dir, prefix)


def generate_narrative_texts(prefix: str) -> None:
    """Generates narrative texts for all splits using LLM and saves to text files.

    Args:
        prefix: Dataset prefix (e.g., 'isic2019') used for manifest naming.
    """
    options_dir = Path("/root/assets/narrative_options")
    prompt_template = (options_dir / "prompt.txt").read_text()

    base_dir = Path("/data/raw")
    manifests_dir = Path("/root/assets/manifests")

    splits = ["train", "val", "test"]
    for split in splits:
        print(f"Starting generation for {split} split...")
        narratives_dir = (
            base_dir
            / ("train" if split in ["train", "val"] else "test")
            / f"{prefix}_narratives"
        )
        narratives_dir.mkdir(parents=True, exist_ok=True)

        narrative_manifest_path = (
            manifests_dir / f"{prefix}_narratives_split_{split}.jsonl"
        )
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

    print(f"\n✓ Narrative text generation complete for {prefix}!")


def verify_narrative_manifests(prefix: str) -> None:
    """Verifies narrative manifests exist and reports symptom distribution statistics.

    Args:
        prefix: Dataset prefix (e.g., 'isic2019') used for manifest naming.
    """
    print(f"\nVerifying {prefix} narrative manifests...")
    manifests_dir = Path("assets/manifests")

    if not manifests_dir.exists():
        print("  ⚠ Manifests directory not found. Run generation first.")
        return

    manifest_files = list(manifests_dir.glob(f"{prefix}_narratives_split_*.jsonl"))

    if not manifest_files:
        print("  ⚠ No narrative manifest files found.")
        return

    all_manifests = {}
    for manifest_file in sorted(manifest_files):
        split_name = manifest_file.stem.replace(f"{prefix}_narratives_split_", "")
        print(f"\n  Verifying {split_name} manifest...")

        entries = []
        with jsonlines.open(manifest_file) as reader:
            entries = list(reader)

        print(f"    Total entries: {len(entries)}")

        missing_texts = sum(1 for e in entries if not Path(e["text_path"]).exists())
        if missing_texts > 0:
            print(f"    ⚠ WARNING: {missing_texts} text files not found")
        else:
            print("    ✓ All text files exist")

        symptom_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for entry in entries:
            num_symptoms = len(entry.get("symptoms", []))

            symptom_counts[num_symptoms] = symptom_counts.get(num_symptoms, 0) + 1

        print("    Symptom count distribution:")
        for num, count in sorted(symptom_counts.items()):
            pct = (count / len(entries)) * 100
            print(f"      {num} symptoms: {count} ({pct:.1f}%)")

        all_manifests[split_name] = entries

    print("\n✓ Verification complete")

    print("\nGenerating distribution visualizations...")
    figures_dir = Path("assets/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _generate_distribution_summaries(all_manifests, figures_dir, prefix)
    print(f"Visualizations saved to {figures_dir}")


def _generate_distribution_summaries(
    manifests: dict[str, list[dict]], output_dir: Path, prefix: str
) -> None:
    """Generates and saves distribution summary plots for narrative manifests.

    Args:
        manifests: Dictionary mapping split names to lists of manifest entries.
        output_dir: Directory to save generated figures.
        prefix: Dataset prefix for output filename.
    """
    _plot_symptom_distribution(manifests, output_dir, prefix)


def _plot_symptom_distribution(
    manifests: dict[str, list[dict]], output_dir: Path, prefix: str
) -> None:
    """Plots symptom count distribution across splits as grouped bar chart.

    Args:
        manifests: Dictionary mapping split names to lists of manifest entries.
        output_dir: Directory to save the generated figure.
        prefix: Dataset prefix for output filename.
    """
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
    plt.savefig(output_dir / f"{prefix}_balance_symptoms.png", dpi=150)
    plt.close()
