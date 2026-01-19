import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep over configs")
    parser.add_argument("--configs", nargs="+", required=True, help="Config paths to run")
    parser.add_argument("--output", default="runs/sweep_summary.csv", help="Summary CSV path")
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> Dict[str, str]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows[-1] if rows else {}


def find_latest_run(output_dir: Path, exp_name: str) -> Path:
    runs = sorted((output_dir / exp_name).glob("*"))
    if not runs:
        raise FileNotFoundError(f"No runs found for {exp_name}")
    return runs[-1]


def main() -> None:
    args = parse_args()
    summary_rows: List[Dict[str, str]] = []
    for config_path in args.configs:
        config = yaml.safe_load(Path(config_path).read_text())
        exp_name = config.get("exp_name", "experiment")
        output_dir = Path(config.get("output_dir", "runs"))
        subprocess.check_call(["python", "-m", "src.train", "--config", config_path])
        run_dir = find_latest_run(output_dir, exp_name)
        metrics = load_metrics(run_dir / "metrics.csv")
        row = {"config": config_path, "run_dir": str(run_dir)}
        row.update(metrics)
        summary_rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
