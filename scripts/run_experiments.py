import argparse
from dataclasses import dataclass, field

import pandas as pd

from build_training_data import PatchDatasetConfig, build_training_dataset
from pipeline_utils import PROCESSED_DIR, REPORTS_DIR
from train_baseline import train_and_evaluate


EXPERIMENT_DATA_DIR = PROCESSED_DIR / "experiments"
EXPERIMENT_REPORT_DIR = REPORTS_DIR / "experiments"


@dataclass(frozen=True)
class ModelExperimentConfig:
    decision_threshold: float = 0.5
    class_weight: str | None = None
    regularization_c: float = 1.0


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    dataset_config: PatchDatasetConfig = field(default_factory=PatchDatasetConfig)
    model_config: ModelExperimentConfig = field(default_factory=ModelExperimentConfig)


PRACTICAL_PLAN = [
    ExperimentSpec(
        name="version_a_threshold_045",
        description="Baseline features with decision threshold 0.45.",
        model_config=ModelExperimentConfig(decision_threshold=0.45),
    ),
    ExperimentSpec(
        name="version_b_balanced_threshold_045",
        description="Balanced logistic regression with decision threshold 0.45.",
        model_config=ModelExperimentConfig(decision_threshold=0.45, class_weight="balanced"),
    ),
    ExperimentSpec(
        name="version_c_clean_labels_balanced",
        description="Clean labels with clear<=0.3, cloudy>=0.7, ambiguous middle dropped, balanced logistic regression.",
        dataset_config=PatchDatasetConfig(clear_threshold=0.3, cloudy_threshold=0.7),
        model_config=ModelExperimentConfig(class_weight="balanced"),
    ),
    ExperimentSpec(
        name="version_d_clean_labels_patch96",
        description="Version C plus 96x96 patches and 64-pixel stride.",
        dataset_config=PatchDatasetConfig(
            patch_size=96,
            patch_stride=64,
            clear_threshold=0.3,
            cloudy_threshold=0.7,
        ),
        model_config=ModelExperimentConfig(class_weight="balanced"),
    ),
]

ALL_RECOMMENDED = [
    ExperimentSpec(
        name="threshold_045",
        description="Baseline data with threshold 0.45.",
        model_config=ModelExperimentConfig(decision_threshold=0.45),
    ),
    ExperimentSpec(
        name="threshold_040",
        description="Baseline data with threshold 0.40.",
        model_config=ModelExperimentConfig(decision_threshold=0.40),
    ),
    ExperimentSpec(
        name="class_weight_balanced",
        description="Baseline data with class_weight='balanced'.",
        model_config=ModelExperimentConfig(class_weight="balanced"),
    ),
    ExperimentSpec(
        name="balanced_threshold_045",
        description="Baseline data with class_weight='balanced' and threshold 0.45.",
        model_config=ModelExperimentConfig(decision_threshold=0.45, class_weight="balanced"),
    ),
    ExperimentSpec(
        name="c_0_1",
        description="Baseline data with regularization C=0.1.",
        model_config=ModelExperimentConfig(regularization_c=0.1),
    ),
    ExperimentSpec(
        name="c_1_0",
        description="Baseline data with regularization C=1.0.",
        model_config=ModelExperimentConfig(regularization_c=1.0),
    ),
    ExperimentSpec(
        name="c_3_0",
        description="Baseline data with regularization C=3.0.",
        model_config=ModelExperimentConfig(regularization_c=3.0),
    ),
    ExperimentSpec(
        name="c_10_0",
        description="Baseline data with regularization C=10.0.",
        model_config=ModelExperimentConfig(regularization_c=10.0),
    ),
    ExperimentSpec(
        name="clean_labels_only",
        description="Labels cleaned to clear<=0.3 and cloudy>=0.7, dropping ambiguous patches.",
        dataset_config=PatchDatasetConfig(clear_threshold=0.3, cloudy_threshold=0.7),
    ),
    ExperimentSpec(
        name="clean_labels_balanced",
        description="Clean labels plus class_weight='balanced'.",
        dataset_config=PatchDatasetConfig(clear_threshold=0.3, cloudy_threshold=0.7),
        model_config=ModelExperimentConfig(class_weight="balanced"),
    ),
    ExperimentSpec(
        name="patch_96_stride_64",
        description="Larger 96x96 patches with 64-pixel stride.",
        dataset_config=PatchDatasetConfig(patch_size=96, patch_stride=64),
    ),
    ExperimentSpec(
        name="patch_128_stride_64",
        description="Larger 128x128 patches with 64-pixel stride.",
        dataset_config=PatchDatasetConfig(patch_size=128, patch_stride=64),
    ),
    ExperimentSpec(
        name="overlap_patch_64_stride_32",
        description="64x64 patches with 32-pixel stride.",
        dataset_config=PatchDatasetConfig(patch_size=64, patch_stride=32),
    ),
    ExperimentSpec(
        name="min_valid_fraction_095",
        description="Baseline patching with min valid fraction 0.95.",
        dataset_config=PatchDatasetConfig(min_valid_fraction=0.95),
    ),
    ExperimentSpec(
        name="min_valid_fraction_080",
        description="Baseline patching with min valid fraction 0.80.",
        dataset_config=PatchDatasetConfig(min_valid_fraction=0.80),
    ),
]

EXPERIMENT_SUITES = {
    "practical_plan": PRACTICAL_PLAN,
    "all_recommended": ALL_RECOMMENDED,
}


def dataset_slug(config):
    clear_value = "none" if config.clear_threshold is None else f"{config.clear_threshold:.2f}".replace(".", "")
    cloudy_value = f"{config.cloudy_threshold:.2f}".replace(".", "")
    valid_value = f"{config.min_valid_fraction:.2f}".replace(".", "")
    return (
        f"ps{config.patch_size}_st{config.patch_stride}"
        f"_vf{valid_value}_clear{clear_value}_cloud{cloudy_value}"
    )


def get_suite(name):
    try:
        return EXPERIMENT_SUITES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown suite '{name}'. Choose from: {', '.join(EXPERIMENT_SUITES)}") from exc


def write_markdown_summary(results_df, summary_path, suite_name):
    top_by_recall = results_df.sort_values(["cloud_recall", "cloud_precision"], ascending=False).iloc[0]
    top_by_f1 = results_df.sort_values(["cloud_f1", "cloud_recall"], ascending=False).iloc[0]
    lines = [
        f"# Experiment results: {suite_name}",
        "",
        f"- Best cloud recall: `{top_by_recall['experiment']}` at `{top_by_recall['cloud_recall']:.4f}` recall and `{top_by_recall['cloud_precision']:.4f}` precision.",
        f"- Best cloud F1: `{top_by_f1['experiment']}` at `{top_by_f1['cloud_f1']:.4f}`.",
        "",
        "| experiment | accuracy | cloud_precision | cloud_recall | cloud_f1 | patches | dropped_ambiguous | threshold | class_weight | C |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    sorted_df = results_df.sort_values(["cloud_recall", "cloud_precision", "accuracy"], ascending=False)
    for _, row in sorted_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['cloud_precision']:.4f}",
                    f"{row['cloud_recall']:.4f}",
                    f"{row['cloud_f1']:.4f}",
                    str(int(row["patches"])),
                    str(int(row["dropped_ambiguous_patches"])),
                    f"{row['decision_threshold']:.2f}",
                    str(row["class_weight"]),
                    f"{row['regularization_c']:.2f}",
                ]
            )
            + " |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(suite_name):
    experiments = get_suite(suite_name)
    dataset_cache = {}
    result_rows = []

    for spec in experiments:
        if spec.dataset_config not in dataset_cache:
            slug = dataset_slug(spec.dataset_config)
            dataset_csv_path = EXPERIMENT_DATA_DIR / f"{slug}.csv"
            dataset_npz_path = EXPERIMENT_DATA_DIR / f"{slug}.npz"
            patch_df, dataset_meta = build_training_dataset(
                config=spec.dataset_config,
                output_csv_path=dataset_csv_path,
                output_npz_path=dataset_npz_path,
                preview_path=None,
            )
            dataset_cache[spec.dataset_config] = {
                "slug": slug,
                "patch_df": patch_df,
                "metadata": dataset_meta,
            }

        dataset_bundle = dataset_cache[spec.dataset_config]
        metrics = train_and_evaluate(
            df=dataset_bundle["patch_df"],
            output_dir=EXPERIMENT_REPORT_DIR,
            report_stem=spec.name,
            summary_title=spec.name.replace("_", " "),
            decision_threshold=spec.model_config.decision_threshold,
            class_weight=spec.model_config.class_weight,
            regularization_c=spec.model_config.regularization_c,
        )

        dataset_meta = dataset_bundle["metadata"]
        result_rows.append(
            {
                "suite": suite_name,
                "experiment": spec.name,
                "description": spec.description,
                "dataset_slug": dataset_bundle["slug"],
                "patches": dataset_meta["patches"],
                "cloudy_patches": dataset_meta["cloudy_patches"],
                "dropped_ambiguous_patches": dataset_meta["dropped_ambiguous_patches"],
                "skipped_empty_scenes": dataset_meta["skipped_empty_scenes"],
                "patch_size": dataset_meta["patch_size"],
                "patch_stride": dataset_meta["patch_stride"],
                "min_valid_fraction": dataset_meta["min_valid_fraction"],
                "clear_threshold": dataset_meta["clear_threshold"],
                "cloudy_threshold": dataset_meta["cloudy_threshold"],
                "accuracy": metrics["accuracy"],
                "cloud_precision": metrics["cloud_precision"],
                "cloud_recall": metrics["cloud_recall"],
                "cloud_f1": metrics["cloud_f1"],
                "decision_threshold": metrics["decision_threshold"],
                "class_weight": metrics["class_weight"],
                "regularization_c": metrics["regularization_c"],
                "split_strategy": metrics["split_strategy"],
                "train_patches": metrics["train_patches"],
                "test_patches": metrics["test_patches"],
            }
        )

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(["cloud_recall", "cloud_precision", "accuracy"], ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"experiment_results_{suite_name}.csv"
    md_path = REPORTS_DIR / f"experiment_results_{suite_name}.md"
    results_df.to_csv(csv_path, index=False)
    write_markdown_summary(results_df, md_path, suite_name)
    return csv_path, md_path, results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run improved-baseline experiment suites.")
    parser.add_argument("--suite", choices=sorted(EXPERIMENT_SUITES), default="practical_plan")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path, md_path, results_df = run_suite(args.suite)
    best_row = results_df.iloc[0]
    print(f"Wrote experiment comparison table to {csv_path}")
    print(f"Wrote experiment markdown summary to {md_path}")
    print(
        "Top recall experiment:",
        best_row["experiment"],
        f"(recall={best_row['cloud_recall']:.4f}, precision={best_row['cloud_precision']:.4f})",
    )
    return 0


if __name__ == "__main__":
    main()
