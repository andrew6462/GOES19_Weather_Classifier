import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import PROCESSED_DIR, RAW_IMAGE_DIR, make_cloud_binary, open_dataset


DEFAULT_PATCH_SIZE = 64
DEFAULT_PATCH_STRIDE = 64
DEFAULT_MIN_VALID_FRACTION = 0.9
DEFAULT_CLOUDY_THRESHOLD = 0.5
FEATURE_NAMES = [
    "cmi_mean",
    "cmi_std",
    "cmi_min",
    "cmi_max",
    "cmi_q25",
    "cmi_median",
    "cmi_q75",
    "valid_fraction",
]


@dataclass(frozen=True)
class PatchDatasetConfig:
    patch_size: int = DEFAULT_PATCH_SIZE
    patch_stride: int = DEFAULT_PATCH_STRIDE
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION
    clear_threshold: Optional[float] = None
    cloudy_threshold: float = DEFAULT_CLOUDY_THRESHOLD

    def validate(self):
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be between 0 and 1")
        if not 0.0 <= self.cloudy_threshold <= 1.0:
            raise ValueError("cloudy_threshold must be between 0 and 1")
        if self.clear_threshold is not None and not 0.0 <= self.clear_threshold <= 1.0:
            raise ValueError("clear_threshold must be between 0 and 1 when provided")
        if self.clear_threshold is not None and self.clear_threshold > self.cloudy_threshold:
            raise ValueError("clear_threshold cannot be greater than cloudy_threshold")


def patch_features(cmi_patch):
    return [
        float(np.mean(cmi_patch)),
        float(np.std(cmi_patch)),
        float(np.min(cmi_patch)),
        float(np.max(cmi_patch)),
        float(np.percentile(cmi_patch, 25)),
        float(np.percentile(cmi_patch, 50)),
        float(np.percentile(cmi_patch, 75)),
    ]


def assign_label(cloud_fraction, config):
    if config.clear_threshold is not None and cloud_fraction <= config.clear_threshold:
        return 0
    if cloud_fraction >= config.cloudy_threshold:
        return 1
    if config.clear_threshold is None:
        return 0
    return None


def build_scene_patches(scene_row, config):
    cmi_ds = open_dataset(scene_row["cmi_path"])
    acm_ds = open_dataset(scene_row["acm_path"])
    cmi_array = np.asarray(cmi_ds["CMI"].values, dtype=np.float32)
    cloud_mask = make_cloud_binary(np.asarray(acm_ds["ACM"].values, dtype=np.float32))
    cmi_ds.close()
    acm_ds.close()

    rows = []
    dropped_ambiguous = 0
    for y0 in range(0, cmi_array.shape[0] - config.patch_size + 1, config.patch_stride):
        for x0 in range(0, cmi_array.shape[1] - config.patch_size + 1, config.patch_stride):
            cmi_patch = cmi_array[y0 : y0 + config.patch_size, x0 : x0 + config.patch_size]
            mask_patch = cloud_mask[y0 : y0 + config.patch_size, x0 : x0 + config.patch_size]

            valid_mask = np.isfinite(cmi_patch) & np.isfinite(mask_patch)
            valid_fraction = float(valid_mask.mean())
            if valid_fraction < config.min_valid_fraction:
                continue

            cmi_valid = cmi_patch[valid_mask]
            cloud_fraction = float(mask_patch[valid_mask].mean())
            label = assign_label(cloud_fraction, config)
            if label is None:
                dropped_ambiguous += 1
                continue

            feature_values = patch_features(cmi_valid)
            rows.append(
                {
                    "patch_id": f"{scene_row['scene_key']}_{y0}_{x0}",
                    "scene_id": scene_row["scene_id"],
                    "scene_key": scene_row["scene_key"],
                    "platform_id": scene_row["platform_id"],
                    "cmi_path": scene_row["cmi_path"],
                    "acm_path": scene_row["acm_path"],
                    "patch_y": y0,
                    "patch_x": x0,
                    "patch_size": config.patch_size,
                    "patch_stride": config.patch_stride,
                    "valid_fraction": valid_fraction,
                    "cloud_fraction": cloud_fraction,
                    "label": label,
                    "storm_overlap": bool(scene_row.get("storm_overlap", False)),
                    "nearest_storm_id": scene_row.get("nearest_storm_id"),
                    "nearest_storm_name": scene_row.get("nearest_storm_name"),
                    "nearest_storm_time_delta_minutes": scene_row.get("nearest_storm_time_delta_minutes"),
                    "cmi_mean": feature_values[0],
                    "cmi_std": feature_values[1],
                    "cmi_min": feature_values[2],
                    "cmi_max": feature_values[3],
                    "cmi_q25": feature_values[4],
                    "cmi_median": feature_values[5],
                    "cmi_q75": feature_values[6],
                }
            )

    return rows, dropped_ambiguous


def save_preview(scene_row, output_path):
    cmi_ds = open_dataset(scene_row["cmi_path"])
    acm_ds = open_dataset(scene_row["acm_path"])
    cmi = np.asarray(cmi_ds["CMI"].values, dtype=np.float32)
    acm = np.asarray(acm_ds["ACM"].values, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cmi, origin="upper")
    axes[0].set_title("CMI")
    axes[1].imshow(acm, origin="upper")
    axes[1].set_title("ACM")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    cmi_ds.close()
    acm_ds.close()


def build_training_dataset(
    config,
    matches_path=PROCESSED_DIR / "matched_scenes.csv",
    output_csv_path=PROCESSED_DIR / "patch_dataset.csv",
    output_npz_path=PROCESSED_DIR / "patch_dataset_arrays.npz",
    preview_path=RAW_IMAGE_DIR / "scene_preview.png",
):
    config.validate()
    matches_df = pd.read_csv(matches_path)

    patch_rows = []
    dropped_ambiguous = 0
    skipped_empty_scenes = 0
    for _, row in matches_df.iterrows():
        scene_rows, scene_dropped = build_scene_patches(row, config)
        if not scene_rows:
            skipped_empty_scenes += 1
            dropped_ambiguous += scene_dropped
            continue
        patch_rows.extend(scene_rows)
        dropped_ambiguous += scene_dropped

    if not patch_rows:
        raise RuntimeError("No labeled patches were produced for this dataset configuration")

    patch_df = pd.DataFrame(patch_rows)
    feature_matrix = patch_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    labels = patch_df["label"].to_numpy(dtype=np.int32)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    patch_df.to_csv(output_csv_path, index=False)
    np.savez(
        output_npz_path,
        X=feature_matrix,
        y=labels,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )

    if preview_path is not None:
        save_preview(matches_df.iloc[0], preview_path)

    metadata = {
        "patches": int(len(patch_df)),
        "cloudy_patches": int(patch_df["label"].sum()),
        "dropped_ambiguous_patches": int(dropped_ambiguous),
        "skipped_empty_scenes": int(skipped_empty_scenes),
        "patch_size": config.patch_size,
        "patch_stride": config.patch_stride,
        "min_valid_fraction": float(config.min_valid_fraction),
        "clear_threshold": config.clear_threshold,
        "cloudy_threshold": float(config.cloudy_threshold),
        "output_csv_path": str(output_csv_path),
        "output_npz_path": str(output_npz_path),
    }
    return patch_df, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build patch-level training data from matched GOES scenes.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--patch-stride", type=int, default=DEFAULT_PATCH_STRIDE)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    parser.add_argument("--clear-threshold", type=float, default=None)
    parser.add_argument("--cloudy-threshold", type=float, default=DEFAULT_CLOUDY_THRESHOLD)
    parser.add_argument("--matches-path", type=Path, default=PROCESSED_DIR / "matched_scenes.csv")
    parser.add_argument("--output-csv-path", type=Path, default=PROCESSED_DIR / "patch_dataset.csv")
    parser.add_argument("--output-npz-path", type=Path, default=PROCESSED_DIR / "patch_dataset_arrays.npz")
    parser.add_argument("--preview-path", type=Path, default=RAW_IMAGE_DIR / "scene_preview.png")
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PatchDatasetConfig(
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        min_valid_fraction=args.min_valid_fraction,
        clear_threshold=args.clear_threshold,
        cloudy_threshold=args.cloudy_threshold,
    )
    preview_path = None if args.skip_preview else args.preview_path
    patch_df, metadata = build_training_dataset(
        config=config,
        matches_path=args.matches_path,
        output_csv_path=args.output_csv_path,
        output_npz_path=args.output_npz_path,
        preview_path=preview_path,
    )
    print(
        "Wrote",
        len(patch_df),
        "patch rows to",
        metadata["output_csv_path"],
        (
            f"(dropped {metadata['dropped_ambiguous_patches']} ambiguous patches, "
            f"skipped {metadata['skipped_empty_scenes']} empty scenes)"
        ),
    )
    return 0


if __name__ == "__main__":
    main()
