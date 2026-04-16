import pandas as pd

from pipeline_utils import INDEX_DIR, PROCESSED_DIR, REPORTS_DIR


def main():
    cmi_df = pd.read_csv(INDEX_DIR / "cmi_index.csv")
    acm_df = pd.read_csv(INDEX_DIR / "acm_index.csv")
    glm_df = pd.read_csv(INDEX_DIR / "glm_index.csv")
    match_df = pd.read_csv(PROCESSED_DIR / "matched_scenes.csv")
    patch_df = pd.read_csv(PROCESSED_DIR / "patch_dataset.csv")
    metrics_df = pd.read_csv(REPORTS_DIR / "baseline_metrics.csv")
    metrics_df["precision_num"] = pd.to_numeric(metrics_df["precision"], errors="coerce")
    metrics_df["recall_num"] = pd.to_numeric(metrics_df["recall"], errors="coerce")
    metrics_df["f1_num"] = pd.to_numeric(metrics_df["f1-score"], errors="coerce")

    positive_row = metrics_df.loc[metrics_df["Unnamed: 0"] == "1"].iloc[0]
    accuracy_row = metrics_df.loc[metrics_df["Unnamed: 0"] == "accuracy"].iloc[0]
    split_row = metrics_df.loc[metrics_df["Unnamed: 0"] == "split_strategy"].iloc[0]
    split_strategy = split_row["precision"]

    report_lines = [
        "# Project Summary",
        "",
        "## Data",
        f"- CMI files: `{len(cmi_df)}`",
        f"- ACM files: `{len(acm_df)}`",
        f"- GLM files: `{len(glm_df)}`",
        f"- matched scenes: `{len(match_df)}`",
        "",
        "## Model setup",
        "- input: CMI patch summary stats",
        "- label: ACM cloud / clear",
        "- model: logistic regression",
        f"- split: `{split_strategy}`",
        "",
        "## Dataset",
        f"- patches: `{len(patch_df)}`",
        f"- cloudy patches: `{int(patch_df['label'].sum())}`",
        f"- storm overlap patches: `{int(patch_df['storm_overlap'].fillna(False).astype(bool).sum())}`",
        "",
        "## Results",
        f"- Accuracy: `{accuracy_row['precision_num']:.4f}`",
        f"- Cloud-class precision: `{positive_row['precision_num']:.4f}`",
        f"- Cloud-class recall: `{positive_row['recall_num']:.4f}`",
        f"- Cloud-class F1: `{positive_row['f1_num']:.4f}`",
        "",
        "## Files",
        "- `reports/confusion_matrix.png`",
        "- `reports/feature_importance.csv`",
        "- `reports/model_summary.md`",
        "- `raw/images/scene_preview.png`",
        "",
        "## Notes",
        "- This grouped-scene run is the frozen baseline for the project.",
        "- The result is more believable than the earlier patch-level fallback run.",
        "- Storm data is still extra context right now, not the main label.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "PROJECT_SUMMARY.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {REPORTS_DIR / 'PROJECT_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    main()
