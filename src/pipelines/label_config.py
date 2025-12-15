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
        "diagnoses": ["melanoma", "mel"],
        "symptoms": "assets/narrative_options/symptoms/mel_symptoms.txt",
    },
    1: {
        "name": "Melanocytic nevus",
        "short": "NV",
        "diagnoses": ["nevus", "nv"],
        "symptoms": "assets/narrative_options/symptoms/nv_symptoms.txt",
    },
    2: {
        "name": "Basal cell carcinoma",
        "short": "BCC",
        "diagnoses": ["basal cell", "bcc"],
        "symptoms": "assets/narrative_options/symptoms/bcc_symptoms.txt",
    },
    3: {
        "name": "Actinic keratosis",
        "short": "AK",
        "diagnoses": ["actinic keratosis", "akiec", "intraepidermal carcinoma"],
        "symptoms": "assets/narrative_options/symptoms/ak_symptoms.txt",
    },
    4: {
        "name": "Benign keratosis",
        "short": "BKL",
        "diagnoses": [
            "benign keratosis",
            "seborrheic keratosis",
            "solar lentigo",
            "lichen planus",
            "bkl",
        ],
        "symptoms": "assets/narrative_options/symptoms/bkl_symptoms.txt",
    },
    5: {
        "name": "Dermatofibroma",
        "short": "DF",
        "diagnoses": ["dermatofibroma", "df"],
        "symptoms": "assets/narrative_options/symptoms/df_symptoms.txt",
    },
    6: {
        "name": "Vascular",
        "short": "VASC",
        "diagnoses": ["vascular", "vasc", "hemangioma", "angioma"],
        "symptoms": "assets/narrative_options/symptoms/vasc_symptoms.txt",
    },
    7: {
        "name": "Squamous cell carcinoma",
        "short": "SCC",
        "diagnoses": ["squamous cell", "scc", "keratoacanthoma"],
        "symptoms": "assets/narrative_options/symptoms/scc_symptoms.txt",
    },
    8: {
        "name": "Inflammatory/infectious",
        "short": "INF",
        "diagnoses": ["inflammatory", "infectious", "inf"],
        "symptoms": "assets/narrative_options/symptoms/inf_symptoms.txt",
    },
    9: {
        "name": "Other benign",
        "short": "BEN_OTH",
        "diagnoses": ["ben_oth", "other benign"],
        "symptoms": "assets/narrative_options/symptoms/ben_oth_symptoms.txt",
    },
    10: {
        "name": "Other malignant",
        "short": "MAL_OTH",
        "diagnoses": ["mal_oth", "other malignant"],
        "symptoms": "assets/narrative_options/symptoms/mal_oth_symptoms.txt",
    },
}

SPLIT_COLORS = {
    "train": "steelblue",
    "val": "coral",
    "test": "lightgreen",
}
