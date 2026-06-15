#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def collect_summaries(input_dir: Path, output_csv: Path, sparse: bool) -> None:
    rows = []
    header = None

    # Recursively find every results_summary.txt in the full tree
    if sparse:
        summary_files = sorted(input_dir.rglob("results_summary_sparse.txt"))
    else:
        summary_files = sorted(input_dir.rglob("results_summary.txt"))

    if not summary_files:
        raise RuntimeError(f"No results_summary.txt files found under {input_dir}")

    for summary_path in summary_files:
        # Use the parent folder name, e.g. "22:50:21.802877"
        result_folder = summary_path.parent.name

        # Optionally also extract the run folder, e.g. backup_full_data_set_no_physics_20-09-44
        # In your structure this is 3 levels above:
        # results_summary.txt -> timestamp -> FNO.0.49.mdlus -> checkpoints -> run_folder
        try:
            run_folder = summary_path.parents[1].name
        except IndexError:
            run_folder = ""

        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            if reader.fieldnames is None:
                print(f"Skipping empty file: {summary_path}")
                continue

            if header is None:
                header = ["run_folder", "result_folder"] + reader.fieldnames

            for row in reader:
                rows.append({
                    "run_folder": run_folder,
                    "result_folder": result_folder,
                    **row,
                })

    if not rows:
        raise RuntimeError("Found summary files, but no data rows were read.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows from {len(summary_files)} summary files to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively combine tab-separated results_summary.txt files into one CSV."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory to search, e.g. test_results_all",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("combined_results_summary.csv"),
        help="Output CSV file path",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="If set, looks for results_summary_sparse.txt instead of results_summary.txt",
    )

    args = parser.parse_args()
    collect_summaries(args.input_dir, args.output, args.sparse)


if __name__ == "__main__":
    main()