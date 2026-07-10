from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .dic2d import run_dic2d
from .mask import run_mask
from .recon3d import run_recon3d
from .scale import run_scale
from .sfm import run_sfm
from .validate import validate_case
from .visualization import run_visualize3d


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
        choices=("validate", "sfm", "scale", "mask", "dic2d", "recon3d", "visualize3d"),
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
        if args.step == "sfm":
            report = run_sfm(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("ok") else 1
        if args.step == "scale":
            report = run_scale(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("ok") else 1
        if args.step == "mask":
            report = run_mask(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("ok") else 1
        if args.step == "dic2d":
            report = run_dic2d(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("ok") else 1
        if args.step == "recon3d":
            report = run_recon3d(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("ok") else 1
        if args.step == "visualize3d":
            report = run_visualize3d(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("ok") else 1

    parser.error("Unsupported command.")
    return 2
