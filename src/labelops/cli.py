"""Small command-line interface for local training, export and benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated integer seeds, such as 7,21,42."
        ) from exc
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed < 0 or seed >= 2**32 for seed in seeds)
    ):
        raise argparse.ArgumentTypeError("Seeds must be distinct integers between 0 and 2**32 - 1.")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="labelops", description="Local human-in-the-loop digit classification workbench."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser(
        "benchmark", help="Run a disclosed oracle-label active-learning simulation."
    )
    benchmark.add_argument("--output", type=Path, default=Path("reports"))
    benchmark.add_argument("--seeds", type=_parse_seeds, default=(7, 21, 42))
    benchmark.add_argument("--rounds", type=int, default=5)
    benchmark.add_argument("--batch-size", type=int, default=40)
    train = commands.add_parser(
        "train", help="Fit the initial labels plus current reviewer consensus."
    )
    train.add_argument("--workspace", type=Path, default=Path(".labelops"))
    export = commands.add_parser("export", help="Export latest reviewer annotations to CSV.")
    export.add_argument("--workspace", type=Path, default=Path(".labelops"))
    export.add_argument("--output", type=Path, default=Path("annotations.csv"))
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            from labelops.benchmark import run_benchmark

            report = run_benchmark(args.output, args.seeds, args.rounds, args.batch_size)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "final_label_budget": report["experiment"]["final_label_budget"],
                        "final_test": report["aggregate"]["final_test"],
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
        elif args.command == "train":
            from labelops.service import Workbench

            summary = Workbench(args.workspace).retrain()
            print(json.dumps(summary, indent=2, allow_nan=False))
        else:
            from labelops.service import Workbench

            annotations = Workbench(args.workspace).store.export_csv()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(annotations, encoding="utf-8")
            print(f"Exported latest reviewer votes to {args.output}")
    except (ValueError, OSError) as exc:
        print(f"labelops: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
