[READ_ME.md](https://github.com/user-attachments/files/26801578/READ_ME.md)
### Modules Used

- `numpy`
- `pandas`
- `matplotlib`
- `xarray`
- `netCDF4`
- `h5netcdf`
- `scikit-learn`
- `tensorflow`

### Data Downloaded

- GOES-19 ABI-L2-CMIPC C13 CONUS files: `529`
- GOES-19 ABI-L2-ACMC CONUS files: `522`
- GOES-19 GLM-L2-LCFA files: `1`
- IBTrACS storm data: `IBTrACS.ALL.v04r01.nc`
- Matched CMI/ACM scenes produced from the indexed files: `521`

### Working Dataset Variants

- Grouped baseline dataset: `461,654` labeled patches
- Final cleaned 96x96 CNN / Version D dataset: `316,466` labeled patches
- Ambiguous patches dropped in the cleaned 96x96 setup: `114,162`

## Baseline

The baseline system used eight summary statistics from each patch and trained a logistic-regression classifier with a grouped scene split.

- Model: `logistic regression`
- Input: `summary statistics from each CMI patch`
- Split strategy: `group_scene_split`
- Train patches: `345,512`
- Test patches: `116,142`
- Decision threshold: `0.50`
- Accuracy: `0.8505`
- Cloud precision: `0.8867`
- Cloud recall: `0.8202`
- Cloud F1: `0.8522`

## Version D

Version D was the strongest feature-based system and became the direct baseline for the final CNN comparison. It kept grouped scene splitting, cleaned the labels, and increased the patch size to `96x96` with stride `64`.

- Model: `logistic regression`
- Dataset rule: `clear <= 0.3`, `cloudy >= 0.7`, ambiguous middle dropped
- Split strategy: `group_scene_split`
- Train patches: `236,741`
- Test patches: `79,725`
- Decision threshold: `0.50`
- Class weight: `balanced`
- Accuracy: `0.9157`
- Cloud precision: `0.9428`
- Cloud recall: `0.8979`
- Cloud F1: `0.9198`

## CNN

The final model replaced hand-crafted patch features with raw `96x96` CMI patches while keeping the same cleaned-label data definition as Version D. Training used a grouped validation split on the training scenes only, early stopping, checkpointing, and validation-based threshold selection.

- Model: `small keras cnn`
- Patch size: `96`
- Patch stride: `64`
- Min valid fraction: `0.9`
- Clear threshold: `0.3`
- Cloudy threshold: `0.7`
- Train patches: `188,792`
- Validation patches: `47,949`
- Test patches: `79,725`
- Batch size: `512`
- Epochs trained: `13`
- Decision threshold selected from validation: `0.45`
- Accuracy: `0.9477`
- Cloud precision: `0.9434`
- Cloud recall: `0.9606`
- Cloud F1: `0.9519`

## Final Comparison

The most important comparison in the project is Version D versus the final CNN because both systems use the same cleaned-label `96x96` patch definition and the same held-out test scenes.

| Model | Accuracy | Cloud Precision | Cloud Recall | Cloud F1 | Test Patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.8505 | 0.8867 | 0.8202 | 0.8522 | 116,142 |
| Version D | 0.9157 | 0.9428 | 0.8979 | 0.9198 | 79,725 |
| Final CNN | 0.9477 | 0.9434 | 0.9606 | 0.9519 | 79,725 |

Compared with Version D, the final CNN improved accuracy by `0.0320`, improved cloud recall by `0.0627`, and improved cloud F1 by `0.0321` on the shared test split. That made the CNN the strongest model produced in the project.
