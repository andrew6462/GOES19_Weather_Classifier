import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR

CNN_PROCESSED_DIR = PROCESSED_DIR / "cnn"
CNN_REPORT_DIR = REPORTS_DIR / "cnn"
DEFAULT_DATASET_PATH = CNN_PROCESSED_DIR / "cnn_dataset_patch96_clean.npz"
DEFAULT_METADATA_PATH = CNN_PROCESSED_DIR / "cnn_dataset_patch96_clean_metadata.csv"
DEFAULT_SUMMARY_PATH = CNN_PROCESSED_DIR / "cnn_dataset_patch96_clean_summary.json"
DEFAULT_MODEL_PATH = CNN_PROCESSED_DIR / "cnn_best_model.keras"

THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


class PatchSequence(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, indices, mean, std, batch_size, shuffle=False):
        self.x_data = x_data
        self.y_data = y_data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch_indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(self.epoch_indices)

    def __len__(self):
        return int(np.ceil(len(self.epoch_indices) / self.batch_size))

    def __getitem__(self, i):
        idx = self.epoch_indices[i * self.batch_size:(i + 1) * self.batch_size]
        x = (self.x_data[idx].astype(np.float32) - self.mean) / self.std
        return x[..., np.newaxis], self.y_data[idx].astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.epoch_indices)
        else:
            self.epoch_indices = self.indices.copy()


def load_inputs(args):
    dataset = np.load(args.dataset_path)
    x_data = dataset["X"]
    y_data = dataset["y"].astype(np.int32)
    metadata_df = pd.read_csv(args.metadata_path)

    if len(metadata_df) != len(x_data):
        raise RuntimeError(f"Metadata has {len(metadata_df)} rows but dataset has {len(x_data)} patches")
    if not np.array_equal(metadata_df["label"].to_numpy(dtype=np.int32), y_data):
        raise RuntimeError("Metadata labels don't match dataset labels")

    summary = None
    if args.dataset_summary_path.exists():
        summary = json.loads(args.dataset_summary_path.read_text())

    return x_data, y_data, metadata_df, summary


def split_data(metadata_df):
    # Split by scene so the same scene can't appear in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_val_idx, test_idx = next(splitter.split(metadata_df, groups=metadata_df["scene_key"]))

    train_val_df = metadata_df.iloc[train_val_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(splitter2.split(train_val_df, groups=train_val_df["scene_key"]))

    splits = {
        "train": train_val_idx[train_idx],
        "val": train_val_idx[val_idx],
        "test": test_idx,
    }

    # Sanity check — scene leakage would silently inflate metrics
    train_scenes = set(metadata_df.iloc[splits["train"]]["scene_key"])
    val_scenes   = set(metadata_df.iloc[splits["val"]]["scene_key"])
    test_scenes  = set(metadata_df.iloc[splits["test"]]["scene_key"])
    if train_scenes & val_scenes or train_scenes & test_scenes or val_scenes & test_scenes:
        raise RuntimeError("Scene leakage detected between splits")

    return splits


def get_normalization_stats(x_data, train_idx):
    # Compute in chunks to avoid loading the whole training set into memory at once
    total = count = sq_total = 0.0
    for i in range(0, len(train_idx), 256):
        batch = x_data[train_idx[i:i+256]].astype(np.float64)
        total    += batch.sum()
        sq_total += (batch ** 2).sum()
        count    += batch.size
    mean = total / count
    std  = max((sq_total / count) - mean ** 2, 1e-12) ** 0.5
    return mean, std


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32,  3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,  3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(x_data, y_data, splits, mean, std, args):
    counts = np.bincount(y_data[splits["train"]], minlength=2)
    # Weight inversely by class frequency to handle cloud/clear imbalance
    class_weight = {0: len(splits["train"]) / (2.0 * counts[0]),
                    1: len(splits["train"]) / (2.0 * counts[1])}

    train_seq = PatchSequence(x_data, y_data, splits["train"], mean, std, args.batch_size, shuffle=True)
    val_seq   = PatchSequence(x_data, y_data, splits["val"],   mean, std, args.batch_size)
    test_seq  = PatchSequence(x_data, y_data, splits["test"],  mean, std, args.batch_size)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model((*x_data.shape[1:3], 1))
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(args.model_path), monitor="val_loss", save_best_only=True),
        ],
        verbose=1,
    )
    model = tf.keras.models.load_model(args.model_path)

    # Pick the decision threshold that maximises cloud F1 on the val set
    val_probs = model.predict(val_seq, verbose=0).ravel()
    scores = {t: f1_score(y_data[splits["val"]], val_probs >= t, zero_division=0) for t in THRESHOLDS}
    threshold, val_f1 = max(scores.items(), key=lambda kv: kv[1])

    test_probs = model.predict(test_seq, verbose=0).ravel()
    test_preds = (test_probs >= threshold).astype(np.int32)

    return history, class_weight, threshold, val_f1, test_preds


def main():
    parser = argparse.ArgumentParser(description="Train a small CNN on GOES CMI patches.")
    parser.add_argument("--dataset-path",        type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--metadata-path",       type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--dataset-summary-path",type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir",          type=Path, default=CNN_REPORT_DIR)
    parser.add_argument("--model-path",          type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--patience",   type=int, default=5)
    parser.add_argument("--baseline-metrics-path", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    x_data, y_data, metadata_df, summary = load_inputs(args)
    splits = split_data(metadata_df)
    mean, std = get_normalization_stats(x_data, splits["train"])

    history, class_weight, threshold, val_f1, test_preds = train(
        x_data, y_data, splits, mean, std, args
    )

    report = classification_report(y_data[splits["test"]], test_preds, output_dict=True, zero_division=0)
    report.update({
        "split_strategy":       "group_scene_split",
        "decision_threshold":   float(threshold),
        "validation_cloud_f1":  float(val_f1),
        "train_patches":        int(len(splits["train"])),
        "val_patches":          int(len(splits["val"])),
        "test_patches":         int(len(splits["test"])),
        "normalization_mean":   float(mean),
        "normalization_std":    float(std),
        "batch_size":           int(args.batch_size),
        "epochs_trained":       int(len(history.history["loss"])),
        "class_weight_0":       float(class_weight[0]),
        "class_weight_1":       float(class_weight[1]),
    })
    if summary:
        for k in ("patch_size", "patch_stride", "min_valid_fraction", "clear_threshold", "cloudy_threshold"):
            if k in summary:
                report[k] = summary[k]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(args.output_dir / "cnn_training_history.csv", index=False)
    (args.output_dir / "cnn_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"Saved model  → {args.model_path}")
    print(f"Wrote report → {args.output_dir / 'cnn_metrics.json'}")


if __name__ == "__main__":
    main()