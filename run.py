from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/MDIC.yaml")
DEFAULT_STEPS = ("validate", "sfm", "scale", "mask", "dic2d", "recon3d", "visualize3d")


def _prefer_local_native_build() -> None:
    """Prefer native extensions built from this checkout over stale editable installs."""

    root = Path(__file__).resolve().parent
    build_root = root / "build"
    candidates = [
        build_root / "wsl-native" / "colmap",
        build_root / "wsl-native" / "recon3d",
        build_root / "windows-native" / "colmap",
        build_root / "windows-native" / "recon3d",
    ]
    if build_root.exists():
        candidates.extend(sorted(build_root.glob("*/colmap")))
        candidates.extend(sorted(build_root.glob("*/recon3d")))
    for candidate in reversed(candidates):
        if candidate.exists():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if "_editable_skbc_pymultidic" not in type(finder).__module__
        and "_editable_skbc_pymultidic" not in repr(finder)
    ]


_prefer_local_native_build()

import pymultidic

STEP_RUNNERS = {
    "validate": pymultidic.run_validate,
    "sfm": pymultidic.run_sfm,
    "scale": pymultidic.run_scale,
    "mask": pymultidic.run_mask,
    "dic2d": pymultidic.run_dic2d,
    "recon3d": pymultidic.run_recon3d,
    "visualize3d": pymultidic.run_visualize3d,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete PyMultiDIC workflow from a YAML config file.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the PyMultiDIC YAML config file.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=list(DEFAULT_STEPS),
        choices=pymultidic.STEP_NAMES,
        help="Workflow steps to run in order.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps after a step reports failure.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for the pipeline JSON report. Defaults to <case_root>/<output_root>/logs/pipeline_report.json.",
    )
    return parser


def _default_report_path(config: Any) -> Path:
    return config.result_root / "logs" / "pipeline_report.json"


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_multidic_flow(
    config_path: str | Path = DEFAULT_CONFIG,
    steps: tuple[str, ...] = DEFAULT_STEPS,
    *,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    config = pymultidic.load_config(config_path, workspace_root=Path.cwd())
    reports: dict[str, Any] = {}
    ok = True

    for step in steps:
        report = STEP_RUNNERS[step](config)
        reports[step] = report
        step_ok = bool(report.get("ok", False))
        ok = ok and step_ok
        if not continue_on_error and not step_ok:
            break

    return {
        "ok": ok,
        "steps": list(steps),
        "completed_steps": list(reports.keys()),
        "reports": reports,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = run_multidic_flow(
        args.config,
        steps=tuple(args.steps),
        continue_on_error=args.continue_on_error,
    )

    config = pymultidic.load_config(args.config, workspace_root=Path.cwd())
    report_path = Path(args.report) if args.report else _default_report_path(config)
    _write_report(report, report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Pipeline report written to: {report_path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
