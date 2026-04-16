import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_utils import REPORTS_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_METRICS_PATH = (
    REPORTS_DIR / "experiments" / "version_d_clean_labels_patch96_baseline_metrics.json"
)


def run_step(script_name, *args):
    script_path = SCRIPT_DIR / script_name
    print("\nRunning", script_name)
    subprocess.run([sys.executable, str(script_path), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the standalone CNN dataset and training pipeline.")
    parser.add_argument("--baseline-metrics-path", type=Path, default=DEFAULT_BASELINE_METRICS_PATH)
    args = parser.parse_args()

    run_step("build_cnn_dataset.py")
    run_step("train_cnn.py", "--baseline-metrics-path", str(args.baseline_metrics_path))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    main()
