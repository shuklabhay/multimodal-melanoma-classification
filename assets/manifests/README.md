# ISIC2019 Manifest Files

## Processing Steps

The manifests were created by applying the following filters to the raw ISIC2019 training dataset:

1. **Filename filtering**: Removed images with `_unknown` or `_downsampled` in filename
2. **Label filtering**: Removed entries with "UNK" or "UNKNOWN" diagnostic labels
3. **Patient-level split**: Split patients into 90% train / 10% validation, ensuring no patient appears in both splits

## Label Mappings

The integer labels in the manifest files correspond to the following diagnoses:

- **Label 0**: MEL (Melanoma)
- **Label 1**: NV (Melanocytic nevus, Nevus)
- **Label 2**: BCC (Basal cell carcinoma)
- **Label 3**: AK (Actinic keratosis, Actinic keratoses)
- **Label 4**: BKL (Benign keratosis, Seborrheic keratosis, Solar lentigo, Lichen planus-like keratosis)
- **Label 5**: DF (Dermatofibroma)
- **Label 6**: VASC (Vascular)
- **Label 7**: SCC (Squamous cell carcinoma)

## Manifest Schema

Each line in the manifest files is a JSON object with the following fields:

```json
{
  "image_path": "/data/raw/train/isic2019/images/ISIC_0000000.jpg",
  "label": 0,
  "patient_id": "IP_1234567",
  "age_approx": 45.0,
  "sex": "male",
  "anatom_site_general": "torso"
}
```
