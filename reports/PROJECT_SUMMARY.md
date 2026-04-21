# Project Summary

## Data
- CMI files: `529`
- ACM files: `522`
- GLM files: `1`
- matched scenes: `521`

## Model setup
- input: CMI patch summary stats
- label: ACM cloud / clear
- model: logistic regression
- split: `group_scene_split`

## Dataset
- patches: `461654`
- cloudy patches: `241234`
- storm overlap patches: `0`

## Results
- Accuracy: `0.8505`
- Cloud-class precision: `0.8867`
- Cloud-class recall: `0.8202`
- Cloud-class F1: `0.8522`

## Files
- `reports/confusion_matrix.png`
- `reports/feature_importance.csv`
- `reports/model_summary.md`
- `raw/images/scene_preview.png`

## Large submission artifacts
- `processed/cnn/cnn_dataset_patch96_clean.npz` (`1.8 GB`, Git LFS): clean `96x96` CNN patch array dataset.
- `processed/cnn/cnn_dataset_patch96_clean_metadata.csv` (`145 MB`, Git LFS): per-patch metadata for the clean CNN dataset.
- `processed/patch_dataset.csv` (`284 MB`, Git LFS): baseline patch-level feature and label table.
- `processed/patch_dataset_arrays.npz` (`16 MB`, Git LFS): baseline patch arrays.
- `processed/cnn/cnn_best_model.keras` (`1.2 MB`, Git LFS): final Keras CNN model.
- See `reports/SUBMISSION_FILES.md` for the full submission inventory.

## Notes
- This grouped-scene run is the frozen baseline for the project.
- The result is more believable than the earlier patch-level fallback run.
- Storm data is still extra context right now, not the main label.
