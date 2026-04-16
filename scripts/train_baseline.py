import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR


FEATURE_COLUMNS = [
    "cmi_mean",
    "cmi_std",
    "cmi_min",
    "cmi_max",
    "cmi_q25",
    "cmi_median",
    "cmi_q75",
    "valid_fraction",
]
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_REGULARIZATION_C = 1.0


def build_split(df):
    scene_count = df["scene_key"].nunique()
    if scene_count >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(df, groups=df["scene_key"]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), "group_scene_split"

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )
    return train_df.copy(), test_df.copy(), "stratified_patch_split_fallback"


def output_path(output_dir, report_stem, filename):
    if report_stem == "baseline":
        return output_dir / filename
    return output_dir / f"{report_stem}_{filename}"


def save_results(y_true, y_pred, split_strategy, output_dir, report_stem, plot_title, metadata):
    report = classification_report(y_true, y_pred, output_dict=True)
    report["split_strategy"] = split_strategy
    report.update(metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_path(output_dir, report_stem, "baseline_metrics.csv"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(output_path(output_dir, report_stem, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    with open(output_path(output_dir, report_stem, "baseline_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def train_and_evaluate(
    df,
    output_dir=REPORTS_DIR,
    report_stem="baseline",
    summary_title="Baseline results",
    decision_threshold=DEFAULT_DECISION_THRESHOLD,
    class_weight=None,
    regularization_c=DEFAULT_REGULARIZATION_C,
    max_iter=1000,
    random_state=42,
):
    train_df, test_df, split_strategy = build_split(df)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    x_test = scaler.transform(test_df[FEATURE_COLUMNS])

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        C=regularization_c,
    )
    model.fit(x_train, train_df["label"])
    positive_probs = model.predict_proba(x_test)[:, 1]
    predictions = (positive_probs >= decision_threshold).astype(np.int32)

    metadata = {
        "decision_threshold": float(decision_threshold),
        "class_weight": class_weight or "none",
        "regularization_c": float(regularization_c),
        "train_patches": int(len(train_df)),
        "test_patches": int(len(test_df)),
    }
    report = save_results(
        test_df["label"].to_numpy(),
        predictions,
        split_strategy,
        output_dir=output_dir,
        report_stem=report_stem,
        plot_title=f"{summary_title} Confusion Matrix",
        metadata=metadata,
    )

    feature_rows = []
    coefficients = model.coef_[0]
    for name, coef in zip(FEATURE_COLUMNS, coefficients):
        feature_rows.append({"feature": name, "coefficient": float(coef)})
    pd.DataFrame(feature_rows).sort_values(
        "coefficient", key=lambda s: s.abs(), ascending=False
    ).to_csv(output_path(output_dir, report_stem, "feature_importance.csv"), index=False)

    summary_lines = [
        f"# {summary_title}",
        "",
        f"- Split strategy: `{split_strategy}`",
        f"- Train patches: `{len(train_df)}`",
        f"- Test patches: `{len(test_df)}`",
        f"- Accuracy: `{report['accuracy']:.4f}`",
        f"- Cloud precision: `{report['1']['precision']:.4f}`",
        f"- Cloud recall: `{report['1']['recall']:.4f}`",
        f"- Cloud F1: `{report['1']['f1-score']:.4f}`",
        f"- Decision threshold: `{decision_threshold:.2f}`",
        f"- Class weight: `{class_weight or 'none'}`",
        f"- Regularization C: `{regularization_c:.2f}`",
    ]
    output_path(output_dir, report_stem, "model_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    return {
        "split_strategy": split_strategy,
        "train_patches": int(len(train_df)),
        "test_patches": int(len(test_df)),
        "accuracy": float(report["accuracy"]),
        "cloud_precision": float(report["1"]["precision"]),
        "cloud_recall": float(report["1"]["recall"]),
        "cloud_f1": float(report["1"]["f1-score"]),
        "decision_threshold": float(decision_threshold),
        "class_weight": class_weight or "none",
        "regularization_c": float(regularization_c),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline cloud classifier.")
    parser.add_argument("--dataset-path", type=Path, default=PROCESSED_DIR / "patch_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--report-stem", type=str, default="baseline")
    parser.add_argument("--summary-title", type=str, default="Baseline results")
    parser.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    parser.add_argument("--class-weight", type=str, default=None)
    parser.add_argument("--regularization-c", type=float, default=DEFAULT_REGULARIZATION_C)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset_path)
    train_and_evaluate(
        df=df,
        output_dir=args.output_dir,
        report_stem=args.report_stem,
        summary_title=args.summary_title,
        decision_threshold=args.decision_threshold,
        class_weight=args.class_weight,
        regularization_c=args.regularization_c,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    print("Wrote baseline files to", args.output_dir)
    return 0


if __name__ == "__main__":
    main()
