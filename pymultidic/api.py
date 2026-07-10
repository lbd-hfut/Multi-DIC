from __future__ import annotations

from pathlib import Path
from typing import Any

from multidic.config import MDICConfig, load_config as _load_config

STEP_NAMES = ("validate", "sfm", "scale", "mask", "dic2d", "recon3d", "visualize3d")


def load_config(config_path: str | Path, workspace_root: str | Path | None = None) -> MDICConfig:
    """Load a Multi-DIC YAML config file."""

    workspace = Path(workspace_root) if workspace_root is not None else None
    return _load_config(Path(config_path), workspace_root=workspace)


def validate_project(config: MDICConfig) -> dict[str, Any]:
    """Validate project inputs and create configured output directories."""

    from multidic.validate import validate_case

    return validate_case(config).to_dict()


def run_validate(config: MDICConfig) -> dict[str, Any]:
    """Alias for :func:`validate_project`."""

    return validate_project(config)


def run_sfm(config: MDICConfig) -> dict[str, Any]:
    from multidic.sfm import run_sfm as _run_sfm

    return _run_sfm(config)


def run_scale(config: MDICConfig) -> dict[str, Any]:
    from multidic.scale import run_scale as _run_scale

    return _run_scale(config)


def run_mask(config: MDICConfig) -> dict[str, Any]:
    from multidic.mask import run_mask as _run_mask

    return _run_mask(config)


def run_dic2d(config: MDICConfig) -> dict[str, Any]:
    from multidic.dic2d import run_dic2d as _run_dic2d

    return _run_dic2d(config)


def run_recon3d(config: MDICConfig) -> dict[str, Any]:
    from multidic.recon3d import run_recon3d as _run_recon3d

    return _run_recon3d(config)


def run_visualize3d(config: MDICConfig) -> dict[str, Any]:
    from multidic.visualization import run_visualize3d as _run_visualize3d

    return _run_visualize3d(config)


def run_step(config: MDICConfig, step: str) -> dict[str, Any]:
    """Run one workflow step by name."""

    step_name = step.lower()
    runners = {
        "validate": run_validate,
        "sfm": run_sfm,
        "scale": run_scale,
        "mask": run_mask,
        "dic2d": run_dic2d,
        "recon3d": run_recon3d,
        "visualize3d": run_visualize3d,
    }
    try:
        return runners[step_name](config)
    except KeyError as exc:
        expected = ", ".join(STEP_NAMES)
        raise ValueError(f"Unknown step {step!r}; expected one of: {expected}") from exc


def run_pipeline(
    config: MDICConfig,
    steps: list[str] | tuple[str, ...] | None = None,
    *,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    """Run a sequence of workflow steps and return a report keyed by step name."""

    selected_steps = tuple(steps) if steps is not None else STEP_NAMES[:-1]
    reports: dict[str, Any] = {}
    ok = True
    for step in selected_steps:
        report = run_step(config, step)
        reports[step] = report
        step_ok = bool(report.get("ok", False))
        ok = ok and step_ok
        if stop_on_error and not step_ok:
            break
    return {
        "ok": ok,
        "steps": list(selected_steps),
        "completed_steps": list(reports.keys()),
        "reports": reports,
    }
