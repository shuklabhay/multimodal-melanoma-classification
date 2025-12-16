import json
import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_utils import use_buffer

app = Typer()


def _run_modal_images(dataset: str, step: str) -> None:
    """Runs Modal images function for specified dataset and step."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::images",
            "--dataset",
            dataset,
            "--step",
            step,
        ],
        check=True,
    )


def _run_modal_create_manifests(dataset: str) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::create_manifests",
            "--dataset",
            dataset,
        ],
        check=True,
    )


def _handle_isic2019_submenu() -> None:
    print("\n=== ISIC2019 Operations ===")
    print("  1. Download images")
    print("  2. Verify download")
    print("  3. Create manifests")
    print("  4. Generate narratives")
    print("  5. Back")
    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        _run_modal_images("isic2019", "download")
    elif choice == "2":
        _run_modal_images("isic2019", "verify_download")
    elif choice == "3":
        _run_modal_create_manifests("isic2019")
        use_buffer(
            {
                "isic2019_split_train.jsonl": "assets/manifests/isic2019_split_train.jsonl",
                "isic2019_split_val.jsonl": "assets/manifests/isic2019_split_val.jsonl",
                "isic2019_split_test.jsonl": "assets/manifests/isic2019_split_test.jsonl",
                "isic2019_balance_label.png": "assets/figures/isic2019_balance_label.png",
                "isic2019_balance_age.png": "assets/figures/isic2019_balance_age.png",
                "isic2019_balance_gender.png": "assets/figures/isic2019_balance_gender.png",
                "isic2019_balance_site.png": "assets/figures/isic2019_balance_site.png",
            }
        )
    elif choice == "4":
        from src.pipelines.dataset_config import ISIC2019_CONFIG
        from src.pipelines.prep_narratives import (
            create_local_narrative_manifests,
            visualize_narrative_manifests,
        )

        create_local_narrative_manifests(ISIC2019_CONFIG)
        visualize_narrative_manifests(ISIC2019_CONFIG)
        subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "run",
                "-m",
                "src.use_modal::narratives_entrypoint",
                "--dataset",
                "isic2019",
            ],
            check=True,
        )


def _handle_milk10k_submenu() -> None:
    print("\n=== MILK10k Operations ===")
    print("  1. Download images")
    print("  2. Verify download")
    print("  3. Create manifests")
    print("  4. Generate narratives")
    print("  5. Back")
    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        _run_modal_images("milk10k", "download")
    elif choice == "2":
        _run_modal_images("milk10k", "verify_download")
    elif choice == "3":
        _run_modal_create_manifests("milk10k")
        use_buffer(
            {
                "milk10k_split_train.jsonl": "assets/manifests/milk10k_split_train.jsonl",
                "milk10k_split_val.jsonl": "assets/manifests/milk10k_split_val.jsonl",
                "milk10k_split_test.jsonl": "assets/manifests/milk10k_split_test.jsonl",
                "milk10k_balance_label.png": "assets/figures/milk10k_balance_label.png",
                "milk10k_balance_age.png": "assets/figures/milk10k_balance_age.png",
                "milk10k_balance_gender.png": "assets/figures/milk10k_balance_gender.png",
                "milk10k_balance_site.png": "assets/figures/milk10k_balance_site.png",
            }
        )
    elif choice == "4":
        from src.pipelines.dataset_config import MILK10K_CONFIG
        from src.pipelines.prep_narratives import (
            create_local_narrative_manifests,
            visualize_narrative_manifests,
        )

        create_local_narrative_manifests(MILK10K_CONFIG)
        visualize_narrative_manifests(MILK10K_CONFIG)
        subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "run",
                "-m",
                "src.use_modal::narratives_entrypoint",
                "--dataset",
                "milk10k",
            ],
            check=True,
        )


def _get_available_sources() -> list[str]:
    """Scans assets/manifests and returns available data sources."""
    manifests_dir = Path("assets/manifests")
    if not manifests_dir.exists():
        return []

    sources = set()
    for f in manifests_dir.glob("*_split_*.jsonl"):
        name = f.stem
        if name.startswith("split_"):
            continue
        if "_narratives_" in name or "_text_embeddings_" in name:
            continue
        source = name.split("_split_")[0]
        sources.add(source)

    return sorted(sources)


def _select_sources(available: list[str], purpose: str) -> list[str]:
    """Prompts user to select data sources."""
    print(f"\n=== Select sources for {purpose} ===")
    for i, source in enumerate(available, 1):
        print(f"  {i}. {source}")
    print(f"  {len(available) + 1}. All")
    print(f"  {len(available) + 2}. Cancel")

    choice = input("\nEnter choices (comma-separated, e.g., 1,2): ").strip()

    if choice == str(len(available) + 2):
        return []
    if choice == str(len(available) + 1):
        return available

    selected = []
    for c in choice.split(","):
        c = c.strip()
        if c.isdigit() and 1 <= int(c) <= len(available):
            selected.append(available[int(c) - 1])

    return selected


def _create_text_embeddings() -> None:
    """Creates text embeddings for selected sources."""
    sources = _get_available_sources()
    if not sources:
        print("No data sources found. Create manifests first.")
        return

    selected = _select_sources(sources, "text embeddings")
    if not selected:
        return

    print(f"\nCreating text embeddings for: {', '.join(selected)}")

    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::embeddings_entrypoint",
        ],
        check=True,
    )

    buffer_map = {}
    for source in selected:
        for split in ["train", "val", "test"]:
            buffer_map[
                f"{source}_text_embeddings_split_{split}.jsonl"
            ] = f"assets/manifests/{source}_text_embeddings_split_{split}.jsonl"

    use_buffer(buffer_map)
    print("\n✓ Text embeddings created!")


def _create_combined_manifests() -> None:
    """Creates combined manifests from selected sources."""
    sources = _get_available_sources()
    if not sources:
        print("No data sources found. Create manifests first.")
        return

    selected = _select_sources(sources, "combined manifests")
    if not selected:
        return

    print(f"\nCreating combined manifests from: {', '.join(selected)}")
    manifests_dir = Path("assets/manifests")

    for split in ["train", "val", "test"]:
        combined_items = []

        for source in selected:
            image_path = manifests_dir / f"{source}_split_{split}.jsonl"
            narrative_path = manifests_dir / f"{source}_narratives_split_{split}.jsonl"
            embedding_path = (
                manifests_dir / f"{source}_text_embeddings_split_{split}.jsonl"
            )

            if not image_path.exists():
                print(f"  ⚠ Missing {image_path.name}, skipping {source}")
                continue

            with open(image_path) as f_img:
                image_lines = f_img.readlines()

            narrative_lines = []
            if narrative_path.exists():
                with open(narrative_path) as f_nar:
                    narrative_lines = f_nar.readlines()

            embedding_lines = []
            if embedding_path.exists():
                with open(embedding_path) as f_emb:
                    embedding_lines = f_emb.readlines()

            for idx, img_line in enumerate(image_lines):
                img_item = json.loads(img_line)

                combined_item = {
                    "image_path": img_item["image_path"],
                    "age_approx": img_item.get("age_approx"),
                    "sex": img_item.get("sex"),
                    "anatom_site_general": img_item.get("anatom_site_general"),
                    "label": img_item["label"],
                    "source": source,
                }

                if idx < len(narrative_lines):
                    nar_item = json.loads(narrative_lines[idx])
                    combined_item["text_path"] = nar_item.get("text_path")

                if idx < len(embedding_lines):
                    emb_item = json.loads(embedding_lines[idx])
                    combined_item["embedding_path"] = emb_item.get("embedding_path")

                combined_items.append(combined_item)

        combined_path = manifests_dir / f"split_{split}.jsonl"
        with open(combined_path, "w") as f_out:
            for item in combined_items:
                f_out.write(json.dumps(item) + "\n")

        print(f"  Created {combined_path.name} with {len(combined_items)} items")

    print("\n✓ Combined manifests created successfully!")


@app.command()
def run_all() -> None:
    """Runs complete data pipeline for all datasets."""
    print("Running complete data pipeline...")


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for data management."""
    if ctx.invoked_subcommand is not None:
        return

    while True:
        print("\n=== Data Management ===")
        print("  1. Handle ISIC2019")
        print("  2. Handle MILK10k")
        print("  3. Create text embeddings (select sources)")
        print("  4. Create combined manifests (select sources)")
        print("  5. Run all")
        print("  6. Exit")
        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            _handle_isic2019_submenu()
        elif choice == "2":
            _handle_milk10k_submenu()
        elif choice == "3":
            _create_text_embeddings()
        elif choice == "4":
            _create_combined_manifests()
        elif choice == "5":
            run_all()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    app()
