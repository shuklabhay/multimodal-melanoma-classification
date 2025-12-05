# multimodal-melanoma-classification

Synthetic patient narrative generation and multimodal melanoma detection.

## Project Structure

```text
lesion-classification/
├── config/           # Training configuration YAML files for different model variants
├── data/             # Data directory (only on Modal volume)
│   ├── raw/          # Raw images/narratives/embeddings (train/test splits)
│   ├── checkpoints/  # Saved weights and metadata
│   └── buffer/       # Temporary file transfer
├── assets/           # Local assets synced to Modal
│   ├── figures/      # Generated figures/results/visualizations
│   ├── manifests/    # Split manifests and patient metadata (JSONL/CSV)
│   └── resources/    # Label mappings, prompts, and templates
├── scripts/          # Entry point scripts
└── src/              # Source code
    ├── pipelines/    # Data preparation pipelines (images, narratives, embeddings)
    ├── models/       # Model architectures (backbones, classifiers, fusion modules)
    └── utils/        # Training, evaluation, and data loading utilities
```
