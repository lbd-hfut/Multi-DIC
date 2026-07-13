from __future__ import annotations

import shutil
import time
import os
import sys
import importlib.machinery
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackendUnavailableError, SfmPaths
from .products import _export_ndef_sfm_products, _select_best_reconstruction, _summarize_reconstructions


class NativeColmapBackend:
    name = "native_colmap"

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def run(self, paths: SfmPaths, image_names: list[str], report: dict) -> list[dict]:
        try:
            native_colmap = self._import_native_colmap()
        except ImportError as exc:
            raise BackendUnavailableError(
                "native_colmap is not available. Build the native extension first, for example: "
                "python -m pip install -e ."
            ) from exc
        if not native_colmap.has_embedded_colmap():
            raise BackendUnavailableError(
                "native_colmap was built without the compact embedded CPU SfM backend. "
                "Reinstall the project so native/colmap/src is compiled."
            )

        self._prepare_workspace(paths)
        text_root = paths.workspace / "colmap_text"

        result = self._run_step(
            report,
            "native_colmap_cpu_sfm",
            native_colmap.run_cpu_sfm,
            str(paths.database_path),
            str(paths.image_root),
            str(paths.sparse_root),
            str(text_root),
            image_names,
            self.options,
        )
        report["native_colmap_capabilities"] = native_colmap.capabilities()
        report["native_colmap_backend"] = result.get("backend", "unknown")
        reconstructions = {
            int(model_id): _read_text_reconstruction(text_root / str(model_id))
            for model_id in result.get("model_ids", [])
        }
        report["command_logs"] = result.get("command_logs", [])
        report["native_colmap_steps"] = result.get("steps", [])
        report["native_colmap_models"] = [
            _model_summary(model_id, reconstruction, paths.camera_names)
            for model_id, reconstruction in sorted(reconstructions.items())
        ]
        summaries = _summarize_reconstructions(reconstructions, paths)
        selected = _select_exportable_reconstruction(reconstructions, paths.camera_names)
        if selected is not None:
            best_model_id, best_reconstruction = selected
            products = _export_ndef_sfm_products(
                best_reconstruction,
                paths,
                image_names,
                reference_camera=str(self.options.get("reference_camera", "cam_0")),
                max_reproj_error=float(self.options.get("max_reproj_error", 4.0)),
                dpi=int(self.options.get("dpi", 180)),
                spatial_outlier_filter=self.options.get("spatial_outlier_filter"),
            )
            report["selected_model_id"] = int(best_model_id)
            report["products"] = products
            report["figures"] = {
                "sparse_scene": products.get("sparse_scene"),
                "camera_observations_3d": products.get("camera_observations_3d"),
                "camera_observations_2d": products.get("camera_observations_2d"),
            }
        elif reconstructions:
            _, best_reconstruction = _select_best_reconstruction(reconstructions)
            registered = sorted(_registered_camera_names(best_reconstruction, paths.camera_names))
            missing = sorted(set(paths.camera_names) - set(registered))
            raise RuntimeError(
                "COLMAP did not produce a model containing every configured camera. "
                f"Best model registered {registered}; missing {missing}. "
                "See report['native_colmap_models'] and report['command_logs'] for diagnostics."
            )
        return summaries

    @staticmethod
    def _add_native_dll_directories() -> None:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "native" / "colmap" / ".deps" / "Library" / "bin",
            root / "native" / "colmap" / ".deps",
        ]
        build_module_dir = root / "build" / "native-colmap-port" / "colmap"
        if any(build_module_dir.glob("native_colmap*.pyd")):
            sys.path.insert(0, str(build_module_dir))
        add_dll_directory = getattr(os, "add_dll_directory", None)
        for candidate in candidates:
            if not candidate.exists():
                continue
            if add_dll_directory is not None:
                add_dll_directory(str(candidate))
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")

    @classmethod
    def _import_native_colmap(cls) -> Any:
        """Prefer the just-built extension over an editable install's stale finder."""
        cls._add_native_dll_directories()
        root = Path(__file__).resolve().parents[2]
        build_module_dir = root / "build" / "native-colmap-port" / "colmap"
        spec = importlib.machinery.PathFinder.find_spec("native_colmap", [str(build_module_dir)])
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules["native_colmap"] = module
            spec.loader.exec_module(module)
            return module
        import native_colmap
        return native_colmap

    def _prepare_workspace(self, paths: SfmPaths) -> None:
        paths.workspace.mkdir(parents=True, exist_ok=True)
        paths.sparse_root.mkdir(parents=True, exist_ok=True)
        if self.options.get("overwrite", True):
            for target in (paths.database_path,):
                if target.exists():
                    target.unlink()
            for target in (paths.sparse_root, paths.workspace / "colmap_text"):
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)

    def _run_step(self, report: dict, name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        step = {"name": name, "ok": False, "elapsed_seconds": None}
        report["steps"].append(step)
        result = func(*args, **kwargs)
        step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        step["ok"] = True
        return result


@dataclass
class _TextCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def calibration_matrix(self) -> np.ndarray:
        model = self.model.upper()
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = self.params[:3]
            return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
            fx, fy, cx, cy = self.params[:4]
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        raise RuntimeError(f"Unsupported COLMAP camera model in text parser: {self.model}")


class _TextRotation:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def matrix(self) -> np.ndarray:
        return self._matrix


@dataclass
class _TextTransform:
    rotation: _TextRotation
    translation: np.ndarray


@dataclass
class _TextPoint2D:
    xy: np.ndarray
    point3D_id: int

    def has_point3D(self) -> bool:
        return self.point3D_id >= 0


@dataclass
class _TextImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    points2D: list[_TextPoint2D]

    @property
    def cam_from_world(self) -> _TextTransform:
        return _TextTransform(_TextRotation(_qvec_to_rotmat(self.qvec)), self.tvec)


@dataclass
class _TextPoint3D:
    point_id: int
    xyz: np.ndarray
    error: float


class _TextReconstruction:
    def __init__(
        self,
        cameras: dict[int, _TextCamera],
        images: dict[int, _TextImage],
        points3D: dict[int, _TextPoint3D],
    ) -> None:
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    @property
    def point3D_ids(self) -> list[int]:
        return sorted(self.points3D)

    def point3D(self, point_id: int) -> _TextPoint3D:
        return self.points3D[int(point_id)]

    def num_images(self) -> int:
        return len(self.images)

    def num_reg_images(self) -> int:
        return len(self.images)

    def num_points3D(self) -> int:
        return len(self.points3D)

    def compute_mean_reprojection_error(self) -> float:
        if not self.points3D:
            return 0.0
        return float(np.mean([point.error for point in self.points3D.values()]))


def _select_exportable_reconstruction(
    reconstructions: dict[int, _TextReconstruction],
    camera_names: tuple[str, ...],
) -> tuple[int, _TextReconstruction] | None:
    ranked = sorted(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
        reverse=True,
    )
    required = set(camera_names)
    for model_id, reconstruction in ranked:
        if required.issubset(_registered_camera_names(reconstruction, camera_names)):
            return model_id, reconstruction
    return None


def _registered_camera_names(reconstruction: _TextReconstruction, camera_names: tuple[str, ...]) -> set[str]:
    registered: set[str] = set()
    for image in reconstruction.images.values():
        for camera_name in camera_names:
            if image.name.startswith(f"{camera_name}_"):
                registered.add(camera_name)
                break
    return registered


def _model_summary(model_id: int, reconstruction: _TextReconstruction, camera_names: tuple[str, ...]) -> dict[str, Any]:
    registered_set = _registered_camera_names(reconstruction, camera_names)
    registered = [name for name in camera_names if name in registered_set]
    missing = [name for name in camera_names if name not in registered_set]
    return {
        "model_id": int(model_id),
        "num_registered_images": reconstruction.num_reg_images(),
        "num_points3D": reconstruction.num_points3D(),
        "mean_reprojection_error": round(reconstruction.compute_mean_reprojection_error(), 6),
        "registered_cameras": registered,
        "missing_cameras": missing,
        "contains_all_configured_cameras": not missing,
    }


def _read_text_reconstruction(model_dir: Path) -> _TextReconstruction:
    return _TextReconstruction(
        cameras=_read_cameras(model_dir / "cameras.txt"),
        images=_read_images(model_dir / "images.txt"),
        points3D=_read_points3d(model_dir / "points3D.txt"),
    )


def _data_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def _read_cameras(path: Path) -> dict[int, _TextCamera]:
    cameras: dict[int, _TextCamera] = {}
    for line in _data_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = _TextCamera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=np.asarray([float(value) for value in parts[4:]], dtype=np.float64),
        )
    return cameras


def _read_images(path: Path) -> dict[int, _TextImage]:
    lines = _data_lines(path)
    images: dict[int, _TextImage] = {}
    for idx in range(0, len(lines), 2):
        header = lines[idx].split()
        points_line = lines[idx + 1].split() if idx + 1 < len(lines) else []
        points: list[_TextPoint2D] = []
        for offset in range(0, len(points_line), 3):
            points.append(
                _TextPoint2D(
                    xy=np.asarray([float(points_line[offset]), float(points_line[offset + 1])], dtype=np.float64),
                    point3D_id=int(points_line[offset + 2]),
                )
            )
        image_id = int(header[0])
        images[image_id] = _TextImage(
            image_id=image_id,
            qvec=np.asarray([float(value) for value in header[1:5]], dtype=np.float64),
            tvec=np.asarray([float(value) for value in header[5:8]], dtype=np.float64),
            camera_id=int(header[8]),
            name=" ".join(header[9:]),
            points2D=points,
        )
    return images


def _read_points3d(path: Path) -> dict[int, _TextPoint3D]:
    points: dict[int, _TextPoint3D] = {}
    for line in _data_lines(path):
        parts = line.split()
        point_id = int(parts[0])
        points[point_id] = _TextPoint3D(
            point_id=point_id,
            xyz=np.asarray([float(value) for value in parts[1:4]], dtype=np.float64),
            error=float(parts[7]),
        )
    return points


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qx * qz + 2.0 * qw * qy],
            [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qw * qx],
            [2.0 * qx * qz - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )
