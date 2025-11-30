import json
import subprocess
import sys
from pathlib import Path

from typer import Context, Typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_utils import use_buffer

app = Typer()


@app.command()
def run_all() -> None:
    """Runs complete data pipeline: download images, process data, and generate narratives."""
    download_images()
    verify_images()
    process_images()
    generate_narrative_manifests()
    generate_narratives()
    create_combined_manifests()


@app.command()
def download_images() -> None:
    """Downloads ISIC 2019 dataset to Modal volume."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::images",
            "--step",
            "download",
        ],
        check=True,
    )


@app.command()
def verify_images() -> None:
    """Verifies downloaded image data integrity on Modal volume."""
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "-m",
            "src.use_modal::images",
            "--step",
            "verify_download",
        ],
        check=True,
    )


@app.command()
def process_images() -> None:
    """Creates image manifests and visualizations, downloads them locally."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::process_images"],
        check=True,
    )
    use_buffer(
        {
            "isic_split_train.jsonl": "assets/manifests/isic_split_train.jsonl",
            "isic_split_val.jsonl": "assets/manifests/isic_split_val.jsonl",
            "isic_split_test.jsonl": "assets/manifests/isic_split_test.jsonl",
            "label_distribution.png": "assets/figures/label_distribution.png",
            "age_distribution.png": "assets/figures/age_distribution.png",
            "sex_distribution.png": "assets/figures/sex_distribution.png",
            "site_distribution.png": "assets/figures/site_distribution.png",
        }
    )


@app.command()
def verify_image_manifests() -> None:
    """Verifies image manifests for data quality (runs locally)."""
    from src.pipelines.prep_images import verify_manifests

    verify_manifests()


@app.command("generate-narratives")
def generate_narratives() -> None:
    """Generates narrative text outputs on Modal using prepared manifests."""
    subprocess.run(
        ["uv", "run", "modal", "run", "-m", "src.use_modal::narratives_entrypoint"],
        check=True,
    )

    print("\n✓ Narrative generation complete!")


@app.command("generate-narrative-manifests")
def generate_narrative_manifests() -> None:
    """Generates local narrative manifests and supporting distribution visualizations."""
    from src.pipelines.prep_narratives import (
        create_local_narrative_manifests,
        visualize_narrative_manifests,
    )

    create_local_narrative_manifests()
    visualize_narrative_manifests()


@app.command()
def verify_narrative_manifests() -> None:
    """Verifies narrative manifests for data quality (runs locally)."""
    from src.pipelines.prep_narratives import verify_narrative_manifests

    verify_narrative_manifests()


@app.command()
def create_combined_manifests() -> None:
    """Creates combined manifests with image_path, text_path, and label from ISIC and narrative manifests."""
    manifests_dir = Path("assets/manifests")

    for split in ["train", "val", "test"]:
        isic_path = manifests_dir / f"isic_split_{split}.jsonl"
        narrative_path = manifests_dir / f"narrative_split_{split}.jsonl"
        combined_path = manifests_dir / f"split_{split}.jsonl"

        print(f"\nCreating combined manifest for {split}...")

        with open(isic_path) as f_isic, open(narrative_path) as f_narrative:
            isic_lines = f_isic.readlines()
            narrative_lines = f_narrative.readlines()

            if len(isic_lines) != len(narrative_lines):
                raise ValueError(
                    f"Mismatched manifest lengths for {split}: "
                    f"ISIC has {len(isic_lines)} items, narrative has {len(narrative_lines)} items"
                )

            combined_items = []
            for idx, (isic_line, narrative_line) in enumerate(
                zip(isic_lines, narrative_lines)
            ):
                isic_item = json.loads(isic_line)
                narrative_item = json.loads(narrative_line)

                if isic_item["label"] != narrative_item["label"]:
                    raise ValueError(
                        f"Label mismatch at line {idx + 1} in {split}: "
                        f"ISIC label={isic_item['label']}, narrative label={narrative_item['label']}"
                    )

                combined_item = {
                    "image_path": isic_item["image_path"],
                    "text_path": narrative_item["text_path"],
                    "age_approx": isic_item["age_approx"],
                    "sex": isic_item["sex"],
                    "anatom_site_general": isic_item["anatom_site_general"],
                    "label": isic_item["label"],
                    "source": "isic",
                }
                combined_items.append(combined_item)

        with open(combined_path, "w") as f_out:
            for item in combined_items:
                f_out.write(json.dumps(item) + "\n")

        print(f"  Created {combined_path.name} with {len(combined_items)} items")

    print("\n✓ Combined manifests created successfully!")


@app.callback(invoke_without_command=True)
def main(ctx: Context) -> None:
    """Interactive CLI for ISIC2019 data management."""
    if ctx.invoked_subcommand is not None:
        return

    print("\n=== ISIC2019 Data Management ===")
    print("What would you like to do?")
    print("  1. Download images")
    print("  2. Verify download")
    print("  3. Create image manifests")
    print("  4. Generate narrative manifests")
    print("  5. Generate narratives")
    print("  6. Create combined manifests")
    print("  7. Run all")
    print("  8. Exit")
    choice = input("\nEnter your choice: ").strip().lower()

    steps_map = {
        "1": download_images,
        "2": verify_images,
        "3": process_images,
        "4": generate_narrative_manifests,
        "5": generate_narratives,
        "6": create_combined_manifests,
        "7": run_all,
    }

    if choice == "8":
        print("Exiting...")
        return

    if "," in choice:
        selected_steps = [s.strip() for s in choice.split(",")]
        invalid_steps = [s for s in selected_steps if s not in steps_map]

        if invalid_steps:
            print(f"Invalid step(s): {', '.join(invalid_steps)}. Please use 1-8.")
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
