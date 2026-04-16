import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from build_training_data import PatchDatasetConfig, assign_label
from pipeline_utils import PROCESSED_DIR, make_cloud_binary, open_dataset


CNN_PROCESSED_DIR = PROCESSED_DIR / "cnn"
DEFAULT_CONFIG = PatchDatasetConfig(
    patch_size=96,
    patch_stride=64,
    min_valid_fraction=0.9,
    clear_threshold=0.3,
    cloudy_threshold=0.7,
)
DEFAULT_DATASET_PATH = CNN_PROCESSED_DIR / "cnn_dataset_patch96_clean.npz"
DEFAULT_METADATA_PATH = CNN_PROCESSED_DIR / "cnn_dataset_patch96_clean_metadata.csv"
DEFAULT_SUMMARY_PATH = CNN_PROCESSED_DIR / "cnn_dataset_patch96_clean_summary.json"


def build_scene_entries(scene_row, config, include_patches=False):
    cmi_ds = open_dataset(scene_row["cmi_path"])
    acm_ds = open_dataset(scene_row["acm_path"])
    cmi_array = np.asarray(cmi_ds["CMI"].values, dtype=np.float32)
    cloud_mask = make_cloud_binary(np.asarray(acm_ds["ACM"].values, dtype=np.float32))
    cmi_ds.close()
    acm_ds.close()

    rows = []
    patches = []
    dropped_ambiguous = 0
    for y0 in range(0, cmi_array.shape[0] - config.patch_size + 1, config.patch_stride):
        for x0 in range(0, cmi_array.shape[1] - config.patch_size + 1, config.patch_stride):
            cmi_patch = cmi_array[y0 : y0 + config.patch_size, x0 : x0 + config.patch_size]
            mask_patch = cloud_mask[y0 : y0 + config.patch_size, x0 : x0 + config.patch_size]

            valid_mask = np.isfinite(cmi_patch) & np.isfinite(mask_patch)
            valid_fraction = float(valid_mask.mean())
            if valid_fraction < config.min_valid_fraction:
                continue

            cloud_fraction = float(mask_patch[valid_mask].mean())
            label = assign_label(cloud_fraction, config)
            if label is None:
                dropped_ambiguous += 1
                continue

            rows.append(
                {
                    "patch_id": f"{scene_row['scene_key']}_{y0}_{x0}",
                    "scene_key": scene_row["scene_key"],
                    "scene_id": scene_row["scene_id"],
                    "cmi_path": scene_row["cmi_path"],
                    "acm_path": scene_row["acm_path"],
                    "patch_y": y0,
                    "patch_x": x0,
                    "patch_size": config.patch_size,
                    "patch_stride": config.patch_stride,
                    "valid_fraction": valid_fraction,
                    "cloud_fraction": cloud_fraction,
                    "label": int(label),
                }
            )

            if include_patches:
                patch_out = np.asarray(cmi_patch, dtype=np.float32).copy()
                if not valid_mask.all():
                    # Keras cannot train on NaNs, so fill missing pixels with the patch mean.
                    fill_value = float(cmi_patch[valid_mask].mean()) if valid_mask.any() else 0.0
                    patch_out[~valid_mask] = fill_value
                patches.append(patch_out.astype(np.float16))

    return rows, patches, dropped_ambiguous


def collect_metadata(matches_df, config):
    metadata_rows = []
    dropped_ambiguous = 0
    skipped_empty_scenes = 0

    for _, row in matches_df.iterrows():
        scene_rows, _, scene_dropped = build_scene_entries(row, config, include_patches=False)
        if not scene_rows:
            skipped_empty_scenes += 1
            dropped_ambiguous += scene_dropped
            continue
        metadata_rows.extend(scene_rows)
        dropped_ambiguous += scene_dropped

    if not metadata_rows:
        raise RuntimeError("No labeled CNN patches were produced for this dataset configuration")

    metadata_df = pd.DataFrame(metadata_rows)
    return metadata_df, dropped_ambiguous, skipped_empty_scenes


def write_patch_array(matches_df, config, output_array_path, expected_count):
    patch_array = open_memmap(
        output_array_path,
        mode="w+",
        dtype=np.float16,
        shape=(expected_count, config.patch_size, config.patch_size),
    )

    next_index = 0
    for _, row in matches_df.iterrows():
        _, scene_patches, _ = build_scene_entries(row, config, include_patches=True)
        if not scene_patches:
            continue
        scene_stack = np.stack(scene_patches, axis=0)
        patch_array[next_index : next_index + len(scene_stack)] = scene_stack
        next_index += len(scene_stack)

    if next_index != expected_count:
        raise RuntimeError(
            f"Patch array count mismatch: expected {expected_count}, wrote {next_index}"
        )

    return patch_array


def build_cnn_dataset(config, matches_path, output_npz_path, metadata_path, summary_path):
    config.validate()
    matches_df = pd.read_csv(matches_path)
    metadata_df, dropped_ambiguous, skipped_empty_scenes = collect_metadata(matches_df, config)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    temp_array_path = output_npz_path.with_suffix(".tmp.npy")
    patch_array = write_patch_array(matches_df, config, temp_array_path, len(metadata_df))
    labels = metadata_df["label"].to_numpy(dtype=np.int32)
    np.savez_compressed(output_npz_path, X=patch_array, y=labels)

    del patch_array
    temp_array_path.unlink(missing_ok=True)

    metadata_df.to_csv(metadata_path, index=False)
    summary = {
        "patches": int(len(metadata_df)),
        "cloudy_patches": int(labels.sum()),
        "dropped_ambiguous_patches": int(dropped_ambiguous),
        "skipped_empty_scenes": int(skipped_empty_scenes),
        "patch_size": int(config.patch_size),
        "patch_stride": int(config.patch_stride),
        "min_valid_fraction": float(config.min_valid_fraction),
        "clear_threshold": config.clear_threshold,
        "cloudy_threshold": float(config.cloudy_threshold),
        "patch_dtype": "float16",
        "invalid_pixel_fill": "patch_mean",
        "dataset_path": str(output_npz_path),
        "metadata_path": str(metadata_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build a raw image patch dataset for CNN training.")
    parser.add_argument("--matches-path", type=Path, default=PROCESSED_DIR / "matched_scenes.csv")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_CONFIG.patch_size)
    parser.add_argument("--patch-stride", type=int, default=DEFAULT_CONFIG.patch_stride)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_CONFIG.min_valid_fraction)
    parser.add_argument("--clear-threshold", type=float, default=DEFAULT_CONFIG.clear_threshold)
    parser.add_argument("--cloudy-threshold", type=float, default=DEFAULT_CONFIG.cloudy_threshold)
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
    summary = build_cnn_dataset(
        config=config,
        matches_path=args.matches_path,
        output_npz_path=args.output_path,
        metadata_path=args.metadata_path,
        summary_path=args.summary_path,
    )
    print(
        "Wrote CNN dataset to",
        args.output_path,
        f"with {summary['patches']} patches",
    )
    return 0


if __name__ == "__main__":
    main()
