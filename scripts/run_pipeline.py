import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script_name, *args):
    script_path = SCRIPT_DIR / script_name
    print("\nRunning", script_name)
    subprocess.run([sys.executable, str(script_path), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end weather classifier pipeline.")
    parser.add_argument("--with-experiments", action="store_true")
    parser.add_argument("--experiment-suite", default="practical_plan")
    args = parser.parse_args()

    scripts = [
        "build_indexes.py",
        "build_matches.py",
        "build_training_data.py",
        "train_baseline.py",
        "generate_report.py",
    ]
    for script_name in scripts:
        run_step(script_name)
    if args.with_experiments:
        run_step("run_experiments.py", "--suite", args.experiment_suite)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    main()
