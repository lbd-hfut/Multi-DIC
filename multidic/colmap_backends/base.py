from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class BackendUnavailableError(RuntimeError):
    """Raised when a configured SfM backend cannot run in this environment."""


@dataclass(frozen=True)
class SfmPaths:
    image_root: Path
    source_image_paths: tuple[Path, ...]
    camera_names: tuple[str, ...]
    workspace: Path
    database_path: Path
    sparse_root: Path
