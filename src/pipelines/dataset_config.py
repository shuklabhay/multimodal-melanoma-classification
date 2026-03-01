from dataclasses import dataclass, field
from typing import Callable

import polars as pl


@dataclass
class DatasetConfig:
    name: str
    train_collection_id: int
    test_collection_id: int
    label_config: dict
    diagnosis_columns: list[str]
    split_by_column: str
    id_column_for_manifest: str
    image_id_column: str
    expected_train_count: int
    expected_test_count: int
    extra_filters: list[Callable[[pl.DataFrame], pl.DataFrame]] = field(
        default_factory=list
    )


ISIC19_LABEL_CONFIG = {
    0: {
        "name": "Melanoma",
        "short": "MEL",
        "diagnoses": ["Melanoma"],
        "symptoms": "assets/narrative_options/symptoms/mel_symptoms.txt",
    },
    1: {
        "name": "Melanocytic nevus",
        "short": "NV",
        "diagnoses": ["Melanocytic nevus", "nevus"],
        "symptoms": "assets/narrative_options/symptoms/nv_symptoms.txt",
    },
    2: {
        "name": "Basal cell carcinoma",
        "short": "BCC",
        "diagnoses": ["Basal cell carcinoma"],
        "symptoms": "assets/narrative_options/symptoms/bcc_symptoms.txt",
    },
    3: {
        "name": "Actinic keratosis",
        "short": "AK",
        "diagnoses": ["Actinic keratosis", "actinic keratoses"],
        "symptoms": "assets/narrative_options/symptoms/ak_symptoms.txt",
    },
    4: {
        "name": "Benign keratosis",
        "short": "BKL",
        "diagnoses": [
            "Benign keratosis",
            "seborrheic keratosis",
            "solar lentigo",
            "lichen planus-like keratosis",
        ],
        "symptoms": "assets/narrative_options/symptoms/bkl_symptoms.txt",
    },
    5: {
        "name": "Dermatofibroma",
        "short": "DF",
        "diagnoses": ["Dermatofibroma"],
        "symptoms": "assets/narrative_options/symptoms/df_symptoms.txt",
    },
    6: {
        "name": "Vascular",
        "short": "VASC",
        "diagnoses": ["Vascular"],
        "symptoms": "assets/narrative_options/symptoms/vasc_symptoms.txt",
    },
    7: {
        "name": "Squamous cell carcinoma",
        "short": "SCC",
        "diagnoses": ["Squamous cell carcinoma"],
        "symptoms": "assets/narrative_options/symptoms/scc_symptoms.txt",
    },
}

ISIC19_SIMILAR_PAIRS = [
    ("MEL", "NV"),
    ("MEL", "BCC"),
    ("MEL", "BKL"),
    ("AK", "SCC"),
    ("AK", "BCC"),
    ("SCC", "BCC"),
]

SPLIT_COLORS = {
    "train": "steelblue",
    "val": "coral",
    "test": "lightgreen",
}


def filter_by_filename(df: pl.DataFrame) -> pl.DataFrame:
    """Filters out rows with isic_id containing '_unknown' or '_downsampled'.

    Args:
        df: Input DataFrame with isic_id column.

    Returns:
        Filtered DataFrame excluding problematic image IDs.
    """
    return df.filter(
        ~pl.col("isic_id").str.contains("_unknown")
        & ~pl.col("isic_id").str.contains("_downsampled")
    )


ISIC2019_CONFIG = DatasetConfig(
    name="isic2019",
    train_collection_id=65,
    test_collection_id=72,
    label_config=ISIC19_LABEL_CONFIG,
    diagnosis_columns=["diagnosis_2", "diagnosis_3"],
    split_by_column="patient_id",
    id_column_for_manifest="patient_id",
    image_id_column="isic_id",
    expected_train_count=25331,
    expected_test_count=8238,
    extra_filters=[filter_by_filename],
)
