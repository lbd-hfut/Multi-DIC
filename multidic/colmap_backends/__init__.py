from __future__ import annotations

from typing import Any

from .base import BackendUnavailableError, ColmapBackend
from .native_colmap_backend import NativeColmapBackend
from .pycolmap_backend import PycolmapBackend


def create_colmap_backend(options: dict[str, Any]) -> ColmapBackend:
    backend_name = str(options.get("backend", "native_colmap")).lower()
    if backend_name in {"native", "native_colmap", "colmap_native"}:
        return NativeColmapBackend(options)
    if backend_name == "pycolmap":
        return PycolmapBackend(options)
    raise BackendUnavailableError(f"Unsupported COLMAP backend: {backend_name}")
