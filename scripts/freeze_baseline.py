from pathlib import Path
import shutil

from pipeline_utils import REPORTS_DIR


FILES_TO_COPY = [
    "baseline_metrics.csv",
    "baseline_metrics.json",
    "feature_importance.csv",
    "model_summary.md",
    "PROJECT_SUMMARY.md",
    "confusion_matrix.png",
]


def main():
    for name in FILES_TO_COPY:
        src = REPORTS_DIR / name
        if not src.exists():
            continue

        if src.suffix:
            frozen_name = src.stem + "_frozen_baseline" + src.suffix
        else:
            frozen_name = src.name + "_frozen_baseline"

        dst = REPORTS_DIR / frozen_name
        shutil.copy2(src, dst)
        print("Saved", dst)

    return 0


if __name__ == "__main__":
    main()
