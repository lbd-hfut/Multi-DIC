from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import MDICConfig
from .image_io import ImageReadError, read_image_size


@dataclass
class CameraReport:
    camera: str
    speckle_frames: list[str]
    calibration_frames: list[str]
    image_size: tuple[int, int] | None


@dataclass
class ValidationReport:
    ok: bool
    project: str
    case_root: str
    result_root: str
    cameras: list[CameraReport] = field(default_factory=list)
    created_directories: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project": self.project,
            "case_root": self.case_root,
            "result_root": self.result_root,
            "camera_count": len(self.cameras),
            "cameras": [
                {
                    "camera": camera.camera,
                    "speckle_frames": camera.speckle_frames,
                    "calibration_frames": camera.calibration_frames,
                    "image_size": list(camera.image_size) if camera.image_size else None,
                }
                for camera in self.cameras
            ],
            "created_directories": self.created_directories,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def validate_case(config: MDICConfig) -> ValidationReport:
    report = ValidationReport(
        ok=False,
        project=config.project.name,
        case_root=str(config.case_root),
        result_root=str(config.result_root),
    )

    _check_directory(config.case_root, "project.case_root", report)
    _check_directory(config.speckle_root, "data.speckle_dir", report)
    _check_directory(config.calibration_root, "data.calibration_dir", report)

    if report.errors:
        return report

    speckle_cameras = _camera_dirs(config.speckle_root, config.data.camera_glob)
    calibration_cameras = _camera_dirs(config.calibration_root, config.data.camera_glob)
    camera_names = sorted(set(speckle_cameras) & set(calibration_cameras), key=_natural_key)

    missing_calibration = sorted(set(speckle_cameras) - set(calibration_cameras), key=_natural_key)
    missing_speckle = sorted(set(calibration_cameras) - set(speckle_cameras), key=_natural_key)
    if missing_calibration:
        report.errors.append(f"Missing calibration camera folders: {missing_calibration}")
    if missing_speckle:
        report.errors.append(f"Missing speckle camera folders: {missing_speckle}")
    if not camera_names:
        report.errors.append("No matching camera folders were found.")

    for camera_name in camera_names:
        camera_report = _validate_camera(config, camera_name, report)
        report.cameras.append(camera_report)

    if not report.errors:
        report.created_directories = _create_output_directories(config)
        report.ok = True
        _write_report(config, report)

    return report


def _validate_camera(config: MDICConfig, camera_name: str, report: ValidationReport) -> CameraReport:
    speckle_dir = config.speckle_root / camera_name
    calibration_dir = config.calibration_root / camera_name
    speckle_frames = _image_files(speckle_dir, config.data.image_extensions)
    calibration_frames = _image_files(calibration_dir, config.data.image_extensions)
    image_size: tuple[int, int] | None = None

    reference = config.data.reference_frame
    if reference not in speckle_frames:
        report.errors.append(f"{camera_name}: missing reference frame '{reference}'.")

    deformed_frames = list(config.data.deformed_frames)
    if not deformed_frames:
        deformed_frames = [frame for frame in speckle_frames if frame != reference]
    if not deformed_frames:
        report.errors.append(f"{camera_name}: no deformed frames were found.")
    for frame in deformed_frames:
        if frame not in speckle_frames:
            report.errors.append(f"{camera_name}: missing deformed frame '{frame}'.")

    if not calibration_frames:
        report.errors.append(f"{camera_name}: no calibration images were found.")

    frames_to_check = [reference, *deformed_frames]
    sizes: dict[str, tuple[int, int]] = {}
    for frame in frames_to_check:
        path = speckle_dir / frame
        if not path.exists():
            continue
        try:
            sizes[frame] = read_image_size(path)
        except ImageReadError as exc:
            report.errors.append(str(exc))

    if sizes:
        image_size = next(iter(sizes.values()))
        mismatched = {name: size for name, size in sizes.items() if size != image_size}
        if mismatched:
            report.errors.append(f"{camera_name}: speckle frame sizes do not match: {mismatched}")

    return CameraReport(
        camera=camera_name,
        speckle_frames=speckle_frames,
        calibration_frames=calibration_frames,
        image_size=image_size,
    )


def _check_directory(path: Path, label: str, report: ValidationReport) -> None:
    if not path.exists():
        report.errors.append(f"{label} does not exist: {path}")
    elif not path.is_dir():
        report.errors.append(f"{label} is not a directory: {path}")


def _camera_dirs(root: Path, pattern: str) -> dict[str, Path]:
    return {
        path.name: path
        for path in root.glob(pattern)
        if path.is_dir()
    }


def _image_files(root: Path, extensions: tuple[str, ...]) -> list[str]:
    files = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=_natural_key)


def _create_output_directories(config: MDICConfig) -> list[str]:
    created: list[str] = []
    config.result_root.mkdir(parents=True, exist_ok=True)
    created.append(str(config.result_root))
    for subdir in config.output.subdirectories:
        path = config.result_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    return created


def _write_report(config: MDICConfig, report: ValidationReport) -> None:
    logs_dir = config.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _natural_key(text: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]
