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

MILK10K_LABEL_CONFIG = {
    0: {
        "name": "Melanoma",
        "short": "MEL",
        "diagnoses": ["melanoma"],
        "symptoms": "assets/narrative_options/symptoms/mel_symptoms.txt",
    },
    1: {
        "name": "Melanocytic nevus",
        "short": "NV",
        "diagnoses": ["nevus"],
        "symptoms": "assets/narrative_options/symptoms/nv_symptoms.txt",
    },
    2: {
        "name": "Basal cell carcinoma",
        "short": "BCC",
        "diagnoses": ["basal cell"],
        "symptoms": "assets/narrative_options/symptoms/bcc_symptoms.txt",
    },
    3: {
        "name": "Actinic keratosis/intraepidermal carcinoma",
        "short": "AKIEC",
        "diagnoses": ["actinic keratosis", "in situ", "bowen"],
        "symptoms": "assets/narrative_options/symptoms/ak_symptoms.txt",
    },
    4: {
        "name": "Benign keratinocytic lesion",
        "short": "BKL",
        "diagnoses": ["acanthoma", "lentigo", "lichen planus", "seborrheic keratosis"],
        "symptoms": "assets/narrative_options/symptoms/bkl_symptoms.txt",
    },
    5: {
        "name": "Dermatofibroma",
        "short": "DF",
        "diagnoses": ["dermatofibroma"],
        "symptoms": "assets/narrative_options/symptoms/df_symptoms.txt",
    },
    6: {
        "name": "Vascular lesions and hemorrhage",
        "short": "VASC",
        "diagnoses": [
            "vascular",
            "angiokeratoma",
            "arterio-venous",
            "hemangioma",
            "lymphangioma",
            "pyogenic granuloma",
        ],
        "symptoms": "assets/narrative_options/symptoms/vasc_symptoms.txt",
    },
    7: {
        "name": "Squamous cell carcinoma/keratoacanthoma",
        "short": "SCCKA",
        "diagnoses": ["sccka", "keratoacanthoma", "squamous cell carcinoma"],
        "symptoms": "assets/narrative_options/symptoms/scc_symptoms.txt",
    },
    8: {
        "name": "Inflammatory and infectious",
        "short": "INF",
        "diagnoses": ["inflammatory", "infectious", "molluscum", "porokeratosis", "verruca"],
        "symptoms": "assets/narrative_options/symptoms/inf_symptoms.txt",
    },
    9: {
        "name": "Other benign proliferations",
        "short": "BEN_OTH",
        "diagnoses": [
            "benign - other",
            "fibro-histiocytic",
            "soft tissue proliferations - vascular",
            "collision - only benign",
            "cylindroma",
            "exogenous",
            "fibroepithelial",
            "fibroma",
            "xanthogranuloma",
            "mastocytosis",
            "melanotic macule",
            "scar",
            "sebaceous hyperplasia",
            "spiradenoma",
            "supernumerary nipple",
            "trichilemmal",
            "trichoblastoma",
        ],
        "symptoms": "assets/narrative_options/symptoms/ben_oth_symptoms.txt",
    },
    10: {
        "name": "Other malignant proliferations",
        "short": "MAL_OTH",
        "diagnoses": [
            "fibroxanthoma",
            "malignant proliferation",
            "kaposi",
            "lymphocytic proliferations",
            "nerve sheath tumor",
            "merkel",
        ],
        "symptoms": "assets/narrative_options/symptoms/mal_oth_symptoms.txt",
    },
}

SPLIT_COLORS = {
    "train": "steelblue",
    "val": "coral",
    "test": "lightgreen",
}


def filter_by_filename(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        ~pl.col("isic_id").str.contains("_unknown")
        & ~pl.col("isic_id").str.contains("_downsampled")
    )


def filter_clinical_only(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("image_type") == "clinical")


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

MILK10K_CONFIG = DatasetConfig(
    name="milk10k",
    train_collection_id=425,
    test_collection_id=424,
    label_config=MILK10K_LABEL_CONFIG,
    diagnosis_columns=["diagnosis_1", "diagnosis_2", "diagnosis_3"],
    split_by_column="lesion_id",
    id_column_for_manifest="lesion_id",
    image_id_column="isic_id",
    expected_train_count=10480,
    expected_test_count=958,
    extra_filters=[filter_clinical_only],
)
