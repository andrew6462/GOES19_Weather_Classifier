# Methods

This project treated cloud detection as a binary patch-classification problem. Starting from matched GOES scenes, the pipeline extracted fixed-size image patches from the GOES-19 ABI Cloud and Moisture Imagery (CMI) product (National Centers for Environmental Information, 2024) and used the paired ABI Level 2 Clear Sky Mask (ACM) product to label each patch as clear or cloudy (National Centers for Environmental Information, 2021). The work was organized in two stages. First, a feature-based baseline was used to establish a reliable data definition and evaluation setup. Second, a convolutional neural network was trained on the raw patches from that same setup. This made it possible to improve the model without changing the underlying task.

## Scene Matching And Patch Labels

The pipeline first matched CMI and ACM files at the scene level. A pair was kept only when the two files came from the same platform and scene, had the same image dimensions, and overlapped in time. This produced 521 matched scenes. The match table also includes Geostationary Lightning Mapper (GLM) metadata (National Centers for Environmental Information, n.d.) and storm metadata, but those fields were retained as context only and were not used as inputs to the final classifier.

For each matched scene, the code slid a square window across the CMI image and the aligned ACM mask. A candidate patch was kept only if at least 90 percent of its pixels were valid in both arrays. The ACM values were reduced to a binary cloud mask, with ACM classes 0 and 1 treated as clear and classes 2 and 3 treated as cloudy. Patch labels were then assigned from the cloud fraction inside the window.

The early baseline used a simple threshold at cloud fraction 0.5. Later experiments adopted a cleaned-label definition that better separated easy clear and easy cloudy cases. Under that rule, a patch was labeled clear only if cloud fraction was at most 0.3, labeled cloudy only if cloud fraction was at least 0.7, and dropped otherwise. This removed ambiguous boundary patches instead of forcing them into one class, which helped limit distortion from uncertain labels (Liu et al., 2024). The final comparable dataset used that cleaned-label rule with 96x96 patches and a stride of 64, which produced 316,466 labeled patches and dropped 114,162 ambiguous ones.

## Baseline Development

The first part of the project used logistic regression so that one change could be tested at a time. Each retained patch was reduced to eight summary features: mean, standard deviation, minimum, maximum, 25th percentile, median, 75th percentile, and valid-pixel fraction. These features were standardized and passed to a logistic-regression classifier implemented in scikit-learn (Scikit-learn developers, n.d.).

Experiments A through D formed the main baseline ladder. Experiment A lowered the decision threshold from 0.50 to 0.45. Experiment B kept that threshold and added class balancing. Experiment C cleaned the labels by using the 0.3 to 0.7 cloud-fraction rule and dropping the middle band. Experiment D kept the cleaned labels and increased patch size from 64x64 to 96x96 with stride 64. Experiment D became the reference baseline for the final comparison because it was the strongest feature-based system and defined the same task later used by the CNN.

## CNN Dataset And Model

The final model kept the Experiment D data definition fixed and replaced hand-crafted features with raw image patches. The CNN dataset stores the full 96x96 CMI patch for each labeled example, along with metadata such as scene key, patch coordinates, valid fraction, and cloud fraction. Because TensorFlow (TensorFlow, 2026) cannot train directly on `NaN` pixels, invalid values were filled with the mean of the valid pixels from the same patch. To keep storage and memory use manageable, the patch array was written through a temporary memory-mapped file and saved in compressed `float16` form.

The network itself was intentionally compact. It used three convolution blocks with 32, 64, and 128 filters, each followed by max pooling. The convolution stack then fed a global average pooling layer, a dense layer with 64 units, a dropout layer with rate 0.20, and a final sigmoid output for binary classification. Training used the Adam optimizer as implemented in TensorFlow (TensorFlow, 2026), binary cross-entropy loss, and binary accuracy.

## Split Strategy And Training Procedure

A central requirement of the project was to avoid leakage across neighboring patches from the same scene (Karasiak et al., 2022). For that reason, all major splits were grouped by `scene_key` instead of performed at random patch level. The held-out test set was created with `GroupShuffleSplit(test_size=0.25, random_state=42)` (Scikit-learn developers, n.d.). A second grouped split was then applied only inside the remaining training scenes to create a validation set for model selection. In the final CNN run, this yielded 188,792 training patches, 47,949 validation patches, and 79,725 test patches.

Normalization statistics were computed from the training split only and applied during batch loading. Class weights were also computed from the training labels so that both classes remained influential during optimization. Training used early stopping, `ReduceLROnPlateau`, and model checkpointing, with batch size 512 and a maximum of 30 epochs.

The final decision threshold was not fixed in advance. After training, the best saved checkpoint was evaluated on the validation scenes at thresholds 0.40, 0.45, 0.50, 0.55, and 0.60. The threshold with the best validation cloud F1 was then used once on the held-out test set. In the latest run, this procedure selected 0.45.

## Why This Method Fit The Project

The final method kept the parts of the project that had already proven useful, especially cleaned labels, larger patches, and scene-level splitting, and changed only the model class. That made the comparison to the best logistic baseline fair. The main expectation was that a CNN trained on raw patches could use texture, edges, and spatial organization that eight summary statistics could not represent. The evaluation section tests whether that added spatial information translated into better held-out performance.
