from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import STEP_NAMES, load_config, run_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pymultidic",
        description="Run the config-driven PyMultiDIC workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one workflow step.")
    run_parser.add_argument(
        "--config",
        default="configs/MDIC.yaml",
        help="Path to the PyMultiDIC YAML config file.",
    )
    run_parser.add_argument(
        "--step",
        default="validate",
        choices=STEP_NAMES,
        help="Workflow step to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(Path(args.config), workspace_root=Path.cwd())
        report = run_step(config, args.step)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    parser.error("Unsupported command.")
    return 2

