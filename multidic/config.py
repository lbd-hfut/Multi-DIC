from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the MDIC config is missing required fields."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    case_root: Path
    output_root: Path


@dataclass(frozen=True)
class DataConfig:
    speckle_dir: str
    calibration_dir: str
    camera_glob: str
    reference_frame: str
    deformed_frames: tuple[str, ...]
    image_extensions: tuple[str, ...]


@dataclass(frozen=True)
class OutputConfig:
    subdirectories: tuple[str, ...]


@dataclass(frozen=True)
class MDICConfig:
    path: Path
    workspace_root: Path
    project: ProjectConfig
    data: DataConfig
    output: OutputConfig
    raw: dict[str, Any]

    @property
    def case_root(self) -> Path:
        return self.project.case_root

    @property
    def result_root(self) -> Path:
        return self.project.case_root / self.project.output_root

    @property
    def speckle_root(self) -> Path:
        return self.project.case_root / self.data.speckle_dir

    @property
    def calibration_root(self) -> Path:
        return self.project.case_root / self.data.calibration_dir


def load_config(config_path: Path, workspace_root: Path | None = None) -> MDICConfig:
    workspace = (workspace_root or Path.cwd()).resolve()
    path = _resolve_existing_path(config_path, workspace)
    raw = _load_yaml(path)

    project_raw = _require_mapping(raw, "project")
    data_raw = _require_mapping(raw, "data")
    output_raw = raw.get("output", {})
    if output_raw is None:
        output_raw = {}
    if not isinstance(output_raw, dict):
        raise ConfigError("Config field 'output' must be a mapping.")

    project_name = _require_str(project_raw, "name")
    case_root = _resolve_workspace_path(_require_str(project_raw, "case_root"), workspace)
    output_root_text = _require_str(project_raw, "output_root")
    output_root = Path(output_root_text)
    if output_root.is_absolute():
        raise ConfigError("project.output_root must be relative to project.case_root.")
    if any(part == ".." for part in output_root.parts):
        raise ConfigError("project.output_root cannot contain '..'.")

    deformed = data_raw.get("deformed_frames", ())
    if deformed is None:
        deformed_frames: tuple[str, ...] = ()
    elif isinstance(deformed, list) and all(isinstance(item, str) for item in deformed):
        deformed_frames = tuple(deformed)
    else:
        raise ConfigError("data.deformed_frames must be a list of file names or null.")

    extensions_raw = data_raw.get("image_extensions", [".bmp", ".png", ".jpg", ".jpeg"])
    if not isinstance(extensions_raw, list) or not all(isinstance(item, str) for item in extensions_raw):
        raise ConfigError("data.image_extensions must be a list of extensions.")

    subdirs_raw = output_raw.get(
        "subdirectories",
        ["logs", "sfm", "scale", "dic2d", "recon3d", "figures"],
    )
    if not isinstance(subdirs_raw, list) or not all(isinstance(item, str) for item in subdirs_raw):
        raise ConfigError("output.subdirectories must be a list of directory names.")

    return MDICConfig(
        path=path,
        workspace_root=workspace,
        project=ProjectConfig(
            name=project_name,
            case_root=case_root,
            output_root=output_root,
        ),
        data=DataConfig(
            speckle_dir=_require_str(data_raw, "speckle_dir"),
            calibration_dir=_require_str(data_raw, "calibration_dir"),
            camera_glob=_require_str(data_raw, "camera_glob"),
            reference_frame=_require_str(data_raw, "reference_frame"),
            deformed_frames=deformed_frames,
            image_extensions=tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions_raw),
        ),
        output=OutputConfig(subdirectories=tuple(subdirs_raw)),
        raw=raw,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to read configs. Install it with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("The config root must be a YAML mapping.")
    return data


def _resolve_existing_path(path: Path, workspace: Path) -> Path:
    resolved = path if path.is_absolute() else workspace / path
    resolved = resolved.resolve()
    if not resolved.exists():
        raise ConfigError(f"Config file does not exist: {resolved}")
    return resolved


def _resolve_workspace_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else workspace / path
    return resolved.resolve()


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config field '{key}' is required and must be a mapping.")
    return value


def _require_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config field '{key}' is required and must be a non-empty string.")
    return value
