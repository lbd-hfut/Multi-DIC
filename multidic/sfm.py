from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .colmap_backends.base import BackendUnavailableError, SfmPaths
from .colmap_backends.native_colmap_backend import NativeColmapBackend
from .config import MDICConfig
from .validate import validate_case


def run_sfm(config: MDICConfig) -> dict[str, Any]:
    validation = validate_case(config)
    report: dict[str, Any] = {
        "ok": False,
        "project": config.project.name,
        "backend": "native_colmap",
        "workspace": str(_sfm_workspace(config)),
        "database_path": str(_database_path(config)),
        "image_path": str(_colmap_image_dir(config)),
        "image_names": [],
        "source_image_names": [],
        "steps": [],
        "reconstructions": [],
        "warnings": list(validation.warnings),
        "errors": list(validation.errors),
    }
    if not validation.ok:
        _write_sfm_report(config, report)
        return report

    try:
        source_image_names, source_image_paths, image_names, camera_names = _prepare_colmap_reference_images(
            config,
            validation.to_dict()["cameras"],
            overwrite=bool(_colmap_config(config).get("overwrite", True)),
        )
    except Exception as exc:
        report["errors"].append(f"Failed to prepare COLMAP reference images: {type(exc).__name__}: {exc}")
        _write_sfm_report(config, report)
        return report

    report["source_image_names"] = source_image_names
    report["image_names"] = image_names
    if len(image_names) < 2:
        report["errors"].append("SfM needs at least two reference images.")
        _write_sfm_report(config, report)
        return report

    paths = SfmPaths(
        image_root=_colmap_image_dir(config),
        source_image_paths=tuple(source_image_paths),
        camera_names=tuple(camera_names),
        workspace=_sfm_workspace(config),
        database_path=_database_path(config),
        sparse_root=_sfm_workspace(config) / "colmap_sfm",
    )
    try:
        backend = NativeColmapBackend(_colmap_config(config))
        report["backend"] = backend.name
        report["reconstructions"] = backend.run(paths, image_names, report)
    except BackendUnavailableError as exc:
        report["errors"].append(str(exc))
        _write_sfm_report(config, report)
        return report
    except Exception as exc:
        report["errors"].append(f"SfM failed: {type(exc).__name__}: {exc}")
        _write_sfm_report(config, report)
        return report

    if report["reconstructions"]:
        report["ok"] = True
    else:
        report["errors"].append("COLMAP completed but produced no reconstruction models.")

    _write_sfm_report(config, report)
    return report


def _prepare_colmap_reference_images(
    config: MDICConfig,
    camera_reports: list[dict[str, Any]],
    overwrite: bool,
) -> tuple[list[str], list[Path], list[str], list[str]]:
    flat_dir = _colmap_image_dir(config)
    if overwrite and flat_dir.exists():
        shutil.rmtree(flat_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)

    source_names: list[str] = []
    source_paths: list[Path] = []
    flat_names: list[str] = []
    camera_names: list[str] = []
    for camera in camera_reports:
        camera_name = camera["camera"]
        speckle_frames = camera.get("speckle_frames") or []
        if not speckle_frames:
            raise ValueError(f"{camera_name}: no speckle images were found.")
        reference_name = str(speckle_frames[0])
        source_rel = Path(config.data.speckle_dir) / camera_name / reference_name
        source_path = config.case_root / source_rel
        flat_name = f"{camera_name}_{reference_name}"
        shutil.copy2(source_path, flat_dir / flat_name)
        source_names.append(source_rel.as_posix())
        source_paths.append(source_path)
        flat_names.append(flat_name)
        camera_names.append(camera_name)
    return source_names, source_paths, flat_names, camera_names


def _write_sfm_report(config: MDICConfig, report: dict[str, Any]) -> None:
    logs_dir = config.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "sfm_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _sfm_workspace(config: MDICConfig) -> Path:
    workspace_name = str(_colmap_config(config).get("workspace", "colmap"))
    return config.result_root / "sfm" / workspace_name


def _database_path(config: MDICConfig) -> Path:
    return _sfm_workspace(config) / "colmap.db"


def _colmap_image_dir(config: MDICConfig) -> Path:
    return _sfm_workspace(config) / "colmap_images"


def _colmap_config(config: MDICConfig) -> dict[str, Any]:
    value = config.raw.get("colmap", {})
    return value if isinstance(value, dict) else {}
