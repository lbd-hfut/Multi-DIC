from __future__ import annotations

from typing import Any

from .base import BackendUnavailableError, ColmapBackend
from .pycolmap_backend import PycolmapBackend


def create_colmap_backend(options: dict[str, Any]) -> ColmapBackend:
    backend_name = str(options.get("backend", "pycolmap")).lower()
    if backend_name == "pycolmap":
        return PycolmapBackend(options)
    raise BackendUnavailableError(f"Unsupported COLMAP backend: {backend_name}")
