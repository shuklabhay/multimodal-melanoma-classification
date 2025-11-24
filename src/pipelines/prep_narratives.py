import asyncio
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import aiofiles
import jsonlines
import matplotlib.pyplot as plt
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm import tqdm

from src.pipelines.label_config import LABEL_CONFIG

STYLE_QUOTAS = {
    "informal": 0.35,
    "formal": 0.15,
    "casual": 0.30,
    "neutral": 0.20,
}

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
        "social_contexts": load_text_file(options_dir / "social_context.txt"),
        "temporal_quantifiers": load_text_file(
            options_dir / "temporal_quantifiers.txt"
        ),
        "prompt_template": (options_dir / "prompt.txt").read_text(),
    }


def categorize_style(style: str) -> str:
    """Categorizes a style string into higher-level groups for quota tracking.

    Args:
        style: Style string to categorize.

    Returns:
        Category name (informal, formal, casual, or neutral).
    """
    informal_keywords = [
        "informal",
        "casual",
        "colloquial",
        "slangy",
        "laidback",
        "relaxed",
        "friendly",
    ]
    formal_keywords = [
        "formal",
        "professional",
        "businesslike",
        "technical",
        "legalistic",
        "scholarly",
    ]
    casual_keywords = [
        "plain",
        "simple",
        "direct",
        "concise",
        "straightforward",
    ]

    style_lower = style.lower()

    if any(kw in style_lower for kw in informal_keywords):
        return "informal"
    elif any(kw in style_lower for kw in formal_keywords):
        return "formal"
    elif any(kw in style_lower for kw in casual_keywords):
        return "casual"
    else:
        return "neutral"


def categorize_memory(memory: str) -> str:
    """Categorizes a memory string into higher-level groups for quota tracking.

    Args:
        memory: Memory vividness description to categorize.

    Returns:
        Category name (vague, approximate, precise, or detailed).
    """
    if "vague" in memory.lower() or "fragmentary" in memory.lower():
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
    symptoms_file = LABEL_CONFIG[label]["symptoms"]
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
    social_context = rng.choice(narrative_options["social_contexts"])
    temporal_quantifier = rng.choice(narrative_options["temporal_quantifiers"])

    return {
        "style": style,
        "memory": memory,
        "social_context": social_context,
        "temporal_quantifier": temporal_quantifier,
        "symptoms": selected_symptoms,
        "style_category": categorize_style(style),
        "memory_category": categorize_memory(memory),
    }


async def generate_narrative_with_nous(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    metadata: dict,
    params: dict,
    prompt_template: str,
) -> str:
    """Makes async API call to Nous (Hermes 4) to generate narrative using OpenAI client.

    Args:
        client: Initialized AsyncOpenAI client.
        limiter: Rate limiter to prevent hitting API limits.
        metadata: Patient metadata (age, sex, site).
        params: Selected narrative parameters.
        prompt_template: Prompt template string.

    Returns:
        Generated narrative text extracted from XML tags.
    """

    meta_json = json.dumps({k: v for k, v in metadata.items() if v is not None})

    clarity_level = "clear" if params["memory_category"] == "precise" else "imprecise"
    symptom_pools = ", ".join(params["symptoms"]) if params["symptoms"] else "none"
    unrelated_detail = ""

    filled_prompt = prompt_template.format(
        meta_json=meta_json,
        clarity_level=clarity_level,
        style_token=params["style"],
        vividness_level=params["memory"],
        symptom_pools=symptom_pools,
        social_context_snippets=params["social_context"],
        temporal_quantifiers=params["temporal_quantifier"],
        unrelated_detail=unrelated_detail,
    )

    async with limiter:
        response = await client.chat.completions.create(
            model="Hermes-4-70B",
            messages=[{"role": "user", "content": filled_prompt}],
            max_tokens=400,
            temperature=0.8,
        )

    raw_response = response.choices[0].message.content
    if not raw_response:
        return ""

    raw_response = raw_response.strip()
    root = ET.fromstring(f"<root>{raw_response}</root>")
    narrative_elem = root.find("narrative")
    if narrative_elem is not None and narrative_elem.text:
        return narrative_elem.text.strip()

    return raw_response


def create_narrative_manifests() -> None:
    """Main orchestrator for narrative generation and manifest creation."""
    print("Starting narrative generation pipeline...")

    continue_from_existing = (
        os.environ.get("CONTINUE_NARRATIVES", "false").lower() == "true"
    )
    if continue_from_existing:
        print("Will continue from where left off")
    else:
        print("Fresh start mode: Will regenerate all narratives")

    options_dir = Path("/root/assets/narrative_options")
    narrative_options = load_narrative_options(options_dir)

    rng = random.Random(42)

    base_dir = Path("/data/raw")
    buffer_dir = Path("/data/buffer")
    buffer_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        print(f"\nProcessing {split} split...")

        narratives_dir = (
            base_dir / ("train" if split == "train" else "test") / "narratives"
        )
        narratives_dir.mkdir(parents=True, exist_ok=True)

        manifest_filename = f"isic_split_{split}.jsonl"
        image_manifest_path = None

        for possible_dir in [
            buffer_dir,
            base_dir / "manifests",
            Path("/root/assets/manifests"),
        ]:
            candidate = possible_dir / manifest_filename
            if candidate.exists():
                image_manifest_path = candidate
                break

        if not image_manifest_path:
            print(f"  Warning: Image manifest not found for {split} split")
            continue

        with jsonlines.open(image_manifest_path) as reader:
            image_entries = list(reader)

        print(f"  Loaded {len(image_entries)} image entries")

        async def generate_batch(
            client: AsyncOpenAI, limiter: AsyncLimiter, batch_entries
        ):
            """Generate narratives for a batch of entries concurrently."""
            tasks = []
            for entry in batch_entries:
                label = entry["label"]
                metadata = {
                    "age_approx": entry.get("age_approx"),
                    "sex": entry.get("sex"),
                    "anatom_site_general": entry.get("anatom_site_general"),
                }
                params = select_narrative_parameters(label, narrative_options, rng)
                tasks.append(
                    (
                        entry,
                        params,
                        generate_narrative_with_nous(
                            client,
                            limiter,
                            metadata,
                            params,
                            narrative_options["prompt_template"],
                        ),
                    )
                )

            entries_list = [t[0] for t in tasks]
            params_list = [t[1] for t in tasks]
            coros = [t[2] for t in tasks]
            narrative_texts = await asyncio.gather(*coros)

            results = []
            for i, text in enumerate(narrative_texts):
                results.append((entries_list[i], params_list[i], text))
            return results

        async def process_all_entries(
            batch_size: int = 4, continue_from_existing: bool = True
        ):
            """Process all entries in batches.

            Args:
                batch_size: Number of concurrent API requests per batch.
                continue_from_existing: If True, skip entries that already have narratives.
            """
            api_key = os.environ.get("NOUS_API_KEY")
            api_url = os.environ.get("NOUS_BASE_URL")

            if not api_key or not api_url:
                raise ValueError(
                    "NOUS_API_KEY and NOUS_BASE_URL must be set in environment"
                )

            client = AsyncOpenAI(api_key=api_key, base_url=api_url)
            limiter = AsyncLimiter(95, 60)
            narrative_entries = []

            entries_to_process = image_entries
            if continue_from_existing:
                existing_count = 0
                entries_to_process = []
                for entry in image_entries:
                    image_path = Path(entry["image_path"])
                    text_filename = image_path.stem + ".txt"
                    text_path = narratives_dir / text_filename
                    if text_path.exists():
                        existing_count += 1
                    else:
                        entries_to_process.append(entry)

                print(
                    f"  Found {existing_count} existing narratives, processing {len(entries_to_process)} remaining"
                )

            if not entries_to_process:
                print("  All narratives already generated, loading from disk...")

            for i in tqdm(
                range(0, len(entries_to_process), batch_size),
                desc=f"Generating {split} narrative batches",
            ):
                batch = entries_to_process[i : i + batch_size]
                results = await generate_batch(client, limiter, batch)

                sample_idx = rng.randint(0, len(results) - 1)

                for idx, (entry, params, narrative_text) in enumerate(results):
                    label = entry["label"]
                    image_path = Path(entry["image_path"])
                    text_filename = image_path.stem + ".txt"
                    text_path = narratives_dir / text_filename

                    if idx == sample_idx:
                        print(narrative_text)

                    async with aiofiles.open(text_path, mode="w") as f:
                        await f.write(narrative_text)

                    narrative_entry = {
                        "text_path": str(text_path),
                        "text": narrative_text,
                        "label": label,
                        "patient_id": entry.get("patient_id", ""),
                        "style": params["style"],
                        "style_category": params["style_category"],
                        "memory": params["memory"],
                        "memory_category": params["memory_category"],
                        "social_context": params["social_context"],
                        "temporal_quantifier": params["temporal_quantifier"],
                        "symptoms": params["symptoms"],
                    }
                    narrative_entries.append(narrative_entry)

            if continue_from_existing:
                print("  Loading existing narratives from disk...")
                for entry in image_entries:
                    image_path = Path(entry["image_path"])
                    text_filename = image_path.stem + ".txt"
                    text_path = narratives_dir / text_filename

                    if text_path.exists():
                        async with aiofiles.open(text_path, mode="r") as f:
                            narrative_text = await f.read()

                        if not any(
                            ne["text_path"] == str(text_path)
                            for ne in narrative_entries
                        ):
                            params = select_narrative_parameters(
                                entry["label"], narrative_options, rng
                            )
                            narrative_entry = {
                                "text_path": str(text_path),
                                "text": narrative_text,
                                "label": entry["label"],
                                "patient_id": entry.get("patient_id", ""),
                                "style": params["style"],
                                "style_category": params["style_category"],
                                "memory": params["memory"],
                                "memory_category": params["memory_category"],
                                "social_context": params["social_context"],
                                "temporal_quantifier": params["temporal_quantifier"],
                                "symptoms": params["symptoms"],
                            }
                            narrative_entries.append(narrative_entry)

            return narrative_entries

        narrative_entries = asyncio.run(
            process_all_entries(continue_from_existing=continue_from_existing)
        )

        manifest_path = buffer_dir / f"narrative_split_{split}.jsonl"
        print(f"  Writing narrative manifest to {manifest_path}...")

        with jsonlines.open(manifest_path, mode="w") as writer:
            writer.write_all(narrative_entries)

        print(f"  Saved {len(narrative_entries)} narrative entries to {manifest_path}")

    print("\nNarrative generation complete!")

    print("\nGenerating distribution summaries...")
    all_manifests = {}
    for split in splits:
        manifest_path = buffer_dir / f"narrative_split_{split}.jsonl"
        if manifest_path.exists():
            with jsonlines.open(manifest_path) as reader:
                all_manifests[split] = list(reader)

    _generate_distribution_summaries(all_manifests, buffer_dir)
    print(f"\nAll outputs saved to {buffer_dir}")


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

        style_counts = {}
        memory_counts = {}
        symptom_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for entry in entries:
            style_cat = entry.get("style_category", "unknown")
            memory_cat = entry.get("memory_category", "unknown")
            num_symptoms = len(entry.get("symptoms", []))

            style_counts[style_cat] = style_counts.get(style_cat, 0) + 1
            memory_counts[memory_cat] = memory_counts.get(memory_cat, 0) + 1
            symptom_counts[num_symptoms] = symptom_counts.get(num_symptoms, 0) + 1

        print(f"    Style distribution:")
        for cat, count in sorted(style_counts.items()):
            pct = (count / len(entries)) * 100
            target = STYLE_QUOTAS.get(cat, 0) * 100
            print(f"      {cat}: {count} ({pct:.1f}%, target: {target:.1f}%)")

        print(f"    Memory distribution:")
        for cat, count in sorted(memory_counts.items()):
            pct = (count / len(entries)) * 100
            target = MEMORY_QUOTAS.get(cat, 0) * 100
            print(f"      {cat}: {count} ({pct:.1f}%, target: {target:.1f}%)")

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
        label_id: config["name"] for label_id, config in LABEL_CONFIG.items()
    }
    _plot_style_distribution(manifests, label_names, output_dir)
    _plot_memory_distribution(manifests, label_names, output_dir)
    _plot_symptom_distribution(manifests, label_names, output_dir)


def _plot_style_distribution(
    manifests: dict[str, list[dict]], label_names: dict, output_dir: Path
) -> None:
    """Plots style category distribution across splits."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    split_colors = {"train": "steelblue", "val": "coral", "test": "lightgreen"}
    split_order = ["train", "val", "test"]

    style_categories = sorted(STYLE_QUOTAS.keys())
    x = range(len(style_categories))
    width = 0.25

    for i, split_name in enumerate(split_order):
        if split_name not in manifests:
            continue

        entries = manifests[split_name]
        total_samples = len(entries)

        style_counts = {}
        for entry in entries:
            style_cat = entry.get("style_category", "neutral")
            style_counts[style_cat] = style_counts.get(style_cat, 0) + 1

        counts = [style_counts.get(cat, 0) for cat in style_categories]
        percentages = [(count / total_samples) * 100 for count in counts]

        offset = width * (i - 1)
        ax.bar(
            [xi + offset for xi in x],
            percentages,
            width,
            label=f"{split_name.capitalize()} (n={total_samples})",
            color=split_colors[split_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [cat.capitalize() for cat in style_categories], rotation=45, ha="right"
    )
    ax.set_xlabel("Style Category")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Style Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "balance_style.png", dpi=150)
    plt.close()


def _plot_memory_distribution(
    manifests: dict[str, list[dict]], label_names: dict, output_dir: Path
) -> None:
    """Plots memory category distribution across splits."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    split_colors = {"train": "steelblue", "val": "coral", "test": "lightgreen"}
    split_order = ["train", "val", "test"]

    memory_categories = sorted(MEMORY_QUOTAS.keys())
    x = range(len(memory_categories))
    width = 0.25

    for i, split_name in enumerate(split_order):
        if split_name not in manifests:
            continue

        entries = manifests[split_name]
        total_samples = len(entries)

        memory_counts = {}
        for entry in entries:
            memory_cat = entry.get("memory_category", "approximate")
            memory_counts[memory_cat] = memory_counts.get(memory_cat, 0) + 1

        counts = [memory_counts.get(cat, 0) for cat in memory_categories]
        percentages = [(count / total_samples) * 100 for count in counts]

        offset = width * (i - 1)
        ax.bar(
            [xi + offset for xi in x],
            percentages,
            width,
            label=f"{split_name.capitalize()} (n={total_samples})",
            color=split_colors[split_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [cat.capitalize() for cat in memory_categories], rotation=45, ha="right"
    )
    ax.set_xlabel("Memory Category")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Memory Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "balance_memory.png", dpi=150)
    plt.close()


def _plot_symptom_distribution(
    manifests: dict[str, list[dict]], label_names: dict, output_dir: Path
) -> None:
    """Plots symptom count distribution across splits."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    split_colors = {"train": "steelblue", "val": "coral", "test": "lightgreen"}
    split_order = ["train", "val", "test"]

    symptom_counts_range = [0, 1, 2, 3]
    x = range(len(symptom_counts_range))
    width = 0.25

    for i, split_name in enumerate(split_order):
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
            color=split_colors[split_name],
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
