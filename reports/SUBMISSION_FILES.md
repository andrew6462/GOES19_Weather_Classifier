# Submission File Inventory

This submission includes generated data, model, and report artifacts for the GOES-19 weather classifier project.

## Large artifacts

These files are intentionally tracked with Git LFS so the repository can include the generated datasets and model without storing them as normal Git blobs.

| Path | Size | Description |
| --- | ---: | --- |
| `processed/cnn/cnn_dataset_patch96_clean.npz` | 1.8 GB | Clean `96x96` CNN patch array dataset with `316,466` patches. |
| `processed/cnn/cnn_dataset_patch96_clean_metadata.csv` | 145 MB | Per-patch metadata for the clean CNN dataset. |
| `processed/patch_dataset.csv` | 284 MB | Baseline patch-level feature and label table. |
| `processed/patch_dataset_arrays.npz` | 16 MB | Baseline patch array export. |
| `processed/cnn/cnn_best_model.keras` | 1.2 MB | Final trained Keras CNN model artifact. |

## Supporting outputs

| Path | Description |
| --- | --- |
| `processed/cnn/cnn_dataset_patch96_clean_summary.json` | CNN dataset generation summary and cleaning thresholds. |
| `processed/matched_scenes.csv` | Matched CMI/ACM scene index used for patch extraction. |
| `reports/cnn/cnn_metrics.json` | Held-out grouped-scene CNN evaluation metrics. |
| `reports/cnn/cnn_training_history.csv` | CNN training history by epoch. |
| `reports/baseline_metrics.csv` | Baseline model metrics. |
| `reports/baseline_metrics.json` | Baseline model metrics in JSON form. |
| `reports/confusion_matrix.png` | Baseline confusion matrix visualization. |
| `reports/feature_importance.csv` | Baseline feature importance table. |
| `reports/model_summary.md` | Baseline model summary. |

## Storage note

The large generated artifacts above require Git LFS when cloning or checking out the repository. Without Git LFS, those paths will appear as pointer files instead of the full datasets or model artifact.
