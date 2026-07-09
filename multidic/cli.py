from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .validate import validate_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m multidic",
        description="Run the config-driven Multi-DIC workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one workflow step.")
    run_parser.add_argument(
        "--config",
        default="configs/MDIC.yaml",
        help="Path to the Multi-DIC YAML config file.",
    )
    run_parser.add_argument(
        "--step",
        default="validate",
        choices=("validate",),
        help="Workflow step to run. More steps will be added incrementally.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(Path(args.config), workspace_root=Path.cwd())
        if args.step == "validate":
            report = validate_case(config)
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0 if report.ok else 1

    parser.error("Unsupported command.")
    return 2
