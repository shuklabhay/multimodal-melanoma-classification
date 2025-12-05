LABEL_CONFIG = {
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

SIMILAR_PAIRS = [
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
