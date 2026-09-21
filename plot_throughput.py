#!/usr/bin/env python3
"""Plot comparable median wall-clock throughput from benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


BUFFER_COLUMN = "buffer_mib"
THROUGHPUT_COLUMN = "median_gatomic_per_s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot median wall-clock atomic-add throughput. If no CSV files are "
            "listed, all CSV files below the repository are plotted."
        )
    )
    parser.add_argument(
        "csv_files",
        metavar="CSV",
        nargs="*",
        type=Path,
        help="benchmark CSV files to plot (default: discover them recursively)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("atomic-add-throughput.png"),
        help="output image path (default: atomic-add-throughput.png)",
    )
    parser.add_argument(
        "--title",
        default="GPU atomic-add throughput",
        help="graph title",
    )
    return parser.parse_args()


def discover_csv_files(script_dir: Path) -> list[Path]:
    return sorted(script_dir.rglob("*.csv"))


def read_series(path: Path) -> tuple[list[float], list[float]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or ())
            required = {BUFFER_COLUMN, THROUGHPUT_COLUMN}
            missing = required - columns
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(f"missing column(s): {missing_names}")

            points: list[tuple[float, float]] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    buffer_mib = float(row[BUFFER_COLUMN])
                    throughput = float(row[THROUGHPUT_COLUMN])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid numeric value on row {row_number}"
                    ) from error
                if buffer_mib <= 0:
                    raise ValueError(
                        f"{BUFFER_COLUMN} must be positive on row {row_number}"
                    )
                points.append((buffer_mib, throughput))
    except OSError as error:
        raise ValueError(str(error)) from error

    if not points:
        raise ValueError("contains no benchmark rows")

    points.sort()
    return [point[0] for point in points], [point[1] for point in points]


def format_buffer_size(buffer_mib: float, _position: float) -> str:
    buffer_bytes = buffer_mib * 1024**2
    if buffer_bytes >= 1024**2:
        value, unit = buffer_bytes / 1024**2, "MiB"
    elif buffer_bytes >= 1024:
        value, unit = buffer_bytes / 1024, "KiB"
    else:
        value, unit = buffer_bytes, "B"
    return f"{value:g} {unit}"


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    csv_files = args.csv_files or discover_csv_files(script_dir)
    if not csv_files:
        print("error: no CSV files found", file=sys.stderr)
        return 2

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator
    except ImportError:
        print(
            "error: matplotlib is required; install it with "
            "'python3 -m pip install matplotlib'",
            file=sys.stderr,
        )
        return 2

    figure, axis = plt.subplots(figsize=(12, 7))
    all_buffer_sizes: list[float] = []
    for csv_path in csv_files:
        try:
            buffer_sizes, throughputs = read_series(csv_path)
        except ValueError as error:
            print(f"error: {csv_path}: {error}", file=sys.stderr)
            return 2
        all_buffer_sizes.extend(buffer_sizes)
        axis.plot(
            buffer_sizes,
            throughputs,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=csv_path.name,
        )

    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=100))
    axis.xaxis.set_major_formatter(FuncFormatter(format_buffer_size))
    axis.set_xlim(min(all_buffer_sizes) * 0.95, max(all_buffer_sizes) * 1.05)
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_xlabel("Buffer size (log2 scale)")
    axis.set_ylabel("Median wall-clock throughput (G atomic adds/s)")
    axis.set_title(args.title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(title="CSV file", fontsize="small", ncols=2)
    figure.tight_layout()

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
    except OSError as error:
        print(f"error: could not write {args.output}: {error}", file=sys.stderr)
        return 2
    finally:
        plt.close(figure)

    print(f"Wrote {args.output} from {len(csv_files)} CSV file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
