from __future__ import annotations

import json
import os
import shutil
import site
import subprocess
import sysconfig
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .config import MDICConfig

MASK_EXTENSIONS = (".npy", ".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def run_dic2d(config: MDICConfig) -> dict[str, Any]:
    dic_raw = config.raw.get("dic2d", {})
    if not isinstance(dic_raw, dict):
        dic_raw = {}
    roi_raw = dic_raw.get("roi", {})
    if not isinstance(roi_raw, dict):
        roi_raw = {}

    paths = _dic2d_paths(config, dic_raw, roi_raw)
    report: dict[str, Any] = {
        "ok": False,
        "project": config.project.name,
        "engine": str(dic_raw.get("engine", "ncorr")),
        "output_dir": str(paths["output_dir"]),
        "errors": [],
        "warnings": [],
        "cameras": [],
        "native_status": "pending",
    }

    try:
        cameras = _load_npz(paths["sfm_dir"] / "cameras.npz")
        observations = _load_npz(paths["sfm_dir"] / "observations.npz")
        cam_names = [str(name) for name in cameras["cam_names"]]
    except Exception as exc:
        report["errors"].append(f"Failed to load SfM outputs: {type(exc).__name__}: {exc}")
        _write_dic2d_report(config, report)
        return report

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    native_dic = _normalize_dic2d_config(dic_raw, roi_raw)
    native_cli = _native_cli_path(config)
    if native_cli is None:
        report["errors"].append(
            "ncorr_cli was not found. Reinstall the project so the native CLI is placed in "
            "pymultidic/bin, or build native/ncorr before running dic2d."
        )
        report["native_status"] = "missing"
        _write_dic2d_report(config, report)
        return report

    for cam_id, cam_name in enumerate(cam_names):
        try:
            roi_path, roi_mode = _resolve_roi_path(config, cam_name, paths["auto_mask_dir"], roi_raw)
            roi_mask = _read_mask(roi_path, int(roi_raw.get("external_threshold", 127)))
            reference_for_seed = _read_gray_image(config.speckle_root / cam_name / config.data.reference_frame)
            observation, seed = _select_seed(observations, cam_id, roi_mask, native_dic, reference_for_seed)
            frames = _run_camera_frames(config, cam_name, roi_mask, observation, seed, native_dic, native_cli, output_dir)
            failed_frames = [frame for frame in frames if not bool(frame.get("ok", False))]
            if failed_frames:
                details = "; ".join(
                    f"{frame.get('deformed_frame')}: {frame.get('error', frame.get('message', 'native CLI failed'))}"
                    for frame in failed_frames
                )
                report["errors"].append(f"{cam_name}: {details}")
            report["cameras"].append(
                {
                    "cam_id": cam_id,
                    "cam_name": cam_name,
                    "roi_mode": roi_mode,
                    "roi_path": str(roi_path),
                    "observation_xy": [float(observation[0]), float(observation[1])],
                    "seed_xy": [float(seed[0]), float(seed[1])],
                    "reference_grid_points": _count_reference_grid_points(roi_mask, native_dic),
                    "roi_min_region_area": int(native_dic.get("roi_min_region_area", 2000)),
                    "native_cli": str(native_cli) if native_cli is not None else None,
                    "frames": frames,
                    "status": "native_error" if failed_frames else "native_affine_grid_dic",
                }
            )
        except Exception as exc:
            report["errors"].append(f"{cam_name}: {type(exc).__name__}: {exc}")

    report["ok"] = not report["errors"]
    report["native_status"] = "affine_grid_dic" if native_cli is not None else "pending"
    report["message"] = (
        "ROI, seed, and per-frame reduced-grid DIC2D outputs are prepared with native six-parameter affine IC-GN refinement."
    )
    _write_dic2d_report(config, report)
    return report


def _dic2d_paths(config: MDICConfig, dic_raw: dict[str, Any], roi_raw: dict[str, Any]) -> dict[str, Path]:
    colmap_cfg = config.raw.get("colmap", {})
    workspace = str(colmap_cfg.get("workspace", "colmap")) if isinstance(colmap_cfg, dict) else "colmap"
    output_name = str(dic_raw.get("output_dir", "dic2d"))
    mask_output = Path(str(roi_raw.get("mask_output_dir", "masks/mask")))
    auto_mask_dir = mask_output if mask_output.is_absolute() else config.result_root / mask_output
    return {
        "sfm_dir": config.result_root / "sfm" / workspace,
        "output_dir": config.result_root / output_name,
        "auto_mask_dir": auto_mask_dir,
    }


def _normalize_dic2d_config(dic_raw: dict[str, Any], roi_raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(dic_raw)
    normalized["roi_min_region_area"] = int(roi_raw.get("min_region_area", dic_raw.get("roi_min_region_area", 2000)))
    fmt = normalized.get("format", {})
    if isinstance(fmt, dict):
        normalized["units_per_pixel"] = float(fmt.get("units_per_pixel", normalized.get("units_per_pixel", 1.0)))
        normalized["cutoff_corrcoef"] = float(fmt.get("cutoff_corrcoef", normalized.get("cutoff_corrcoef", 0.6)))
        normalized["lenscoef"] = float(fmt.get("lenscoef", normalized.get("lenscoef", 0.0)))
    return normalized


def _native_cli_path(config: MDICConfig) -> Path | None:
    exe_name = "ncorr_cli.exe" if os.name == "nt" else "ncorr_cli"
    candidates: list[Path] = []
    try:
        candidates.append(Path(str(files("pymultidic").joinpath("bin", exe_name))))
    except Exception:
        pass
    command_path = shutil.which(exe_name)
    if command_path:
        candidates.append(Path(command_path))
    candidates.extend(
        [
            Path(site.getusersitepackages()) / "pymultidic" / "bin" / exe_name,
            Path(sysconfig.get_path("purelib")) / "pymultidic" / "bin" / exe_name,
        ]
    )
    candidates.extend(
        [
            config.workspace_root / "build" / "wsl-native" / "ncorr" / "ncorr_cli",
            config.workspace_root / "build" / "wsl-fulltest" / "ncorr" / "ncorr_cli",
            config.workspace_root / "native" / "ncorr" / "build" / "ncorr_cli",
            config.workspace_root / "native" / "ncorr" / "build" / exe_name,
        ]
    )
    build_root = config.workspace_root / "build"
    if build_root.exists():
        candidates.extend(sorted(build_root.glob(f"*/ncorr/{exe_name}")))
        if os.name != "nt":
            candidates.extend(sorted(build_root.glob("*/ncorr/ncorr_cli")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_camera_frames(
    config: MDICConfig,
    cam_name: str,
    roi_mask: np.ndarray,
    observation: np.ndarray,
    seed: np.ndarray,
    dic_raw: dict[str, Any],
    native_cli: Path | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    cam_dir = config.speckle_root / cam_name
    reference_path = cam_dir / config.data.reference_frame
    reference = _read_gray_image(reference_path)
    frames: list[dict[str, Any]] = []
    for frame_name in config.data.deformed_frames:
        deformed_path = cam_dir / frame_name
        frame_report: dict[str, Any] = {
            "reference_frame": config.data.reference_frame,
            "deformed_frame": frame_name,
            "reference_path": str(reference_path),
            "deformed_path": str(deformed_path),
        }
        if native_cli is None:
            frame_report["status"] = "native_cli_missing"
        else:
            deformed = _read_gray_image(deformed_path)
            result_path = output_dir / f"dic2d_{cam_name}_{Path(frame_name).stem}.npz"
            if result_path.exists():
                result_path.unlink()
            frame_report.update(
                _call_native_cli(native_cli, reference, deformed, roi_mask, observation, seed, dic_raw, result_path)
            )
        frames.append(frame_report)
    return frames


def _read_gray_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim == 3:
            data = data[..., 0]
        return np.asarray(data, dtype=np.float64)
    import cv2

    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise ValueError(f"Could not read image: {path}")
    return np.asarray(data, dtype=np.float64)


def _call_native_cli(
    native_cli: Path,
    reference: np.ndarray,
    deformed: np.ndarray,
    roi_mask: np.ndarray,
    observation: np.ndarray,
    seed: np.ndarray,
    dic_raw: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    if reference.shape != deformed.shape or reference.shape != roi_mask.shape[:2]:
        raise ValueError("Reference, deformed, and ROI mask dimensions must match.")
    with tempfile.TemporaryDirectory(prefix="mdic_ncorr_") as temp_name:
        temp_dir = Path(temp_name)
        reference_bin = temp_dir / "reference.bin"
        deformed_bin = temp_dir / "deformed.bin"
        mask_bin = temp_dir / "mask.bin"
        output_bin = temp_dir / "result.bin"
        _write_gray_binary(reference_bin, reference)
        _write_gray_binary(deformed_bin, deformed)
        _write_mask_binary(mask_bin, roi_mask)
        cmd = _native_command(native_cli, reference_bin, deformed_bin, mask_bin, observation, seed, dic_raw, output_bin)
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        result_arrays = _read_result_binary(output_bin) if output_bin.exists() else None
    payload = json.loads(proc.stdout or "{}")
    if result_arrays is not None:
        result_arrays["subset_radius"] = np.asarray(int(dic_raw.get("subset_radius", 20)), dtype=np.int32)
        result_arrays["subset_spacing"] = np.asarray(int(dic_raw.get("subset_spacing", 5)), dtype=np.int32)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_path, **result_arrays)
        payload["output_npz"] = str(result_path)
        valid = np.asarray(result_arrays["valid"], dtype=bool)
        corrcoef = np.asarray(result_arrays["corrcoef"], dtype=np.float64)
        valid_corr = corrcoef[valid]
        payload["valid_ratio"] = float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0
        payload["mean_corrcoef"] = float(np.mean(valid_corr)) if valid_corr.size else 0.0
        payload["median_corrcoef"] = float(np.median(valid_corr)) if valid_corr.size else 0.0
        payload["failed_points"] = int(valid.size - np.count_nonzero(valid))
        payload["output_schema_version"] = int(payload.get("output_schema_version", 2))
    payload["status"] = "native_affine_grid_dic" if payload.get("ok") else "native_error"
    payload["returncode"] = proc.returncode
    if proc.returncode != 0 and proc.stderr and proc.stderr.strip():
        payload["stderr"] = proc.stderr.strip().encode("ascii", errors="replace").decode("ascii")
    return payload


def _native_command(
    native_cli: Path,
    reference_bin: Path,
    deformed_bin: Path,
    mask_bin: Path,
    observation: np.ndarray,
    seed: np.ndarray,
    dic_raw: dict[str, Any],
    output_bin: Path,
) -> list[str]:
    args = [
        str(native_cli),
        "--reference",
        str(reference_bin),
        "--deformed",
        str(deformed_bin),
        "--mask",
        str(mask_bin),
        "--output",
        str(output_bin),
        "--seed-x",
        str(float(seed[0])),
        "--seed-y",
        str(float(seed[1])),
        "--obs-x",
        str(float(observation[0])),
        "--obs-y",
        str(float(observation[1])),
        "--subset-radius",
        str(int(dic_raw.get("subset_radius", 20))),
        "--subset-spacing",
        str(int(dic_raw.get("subset_spacing", 5))),
        "--seed-search-radius",
        str(int(dic_raw.get("seed_search_radius", 50))),
        "--rg-search-radius",
        str(int(dic_raw.get("rg_search_radius", 1))),
        "--roi-min-region-area",
        str(int(dic_raw.get("roi_min_region_area", 2000))),
        "--cutoff-diffnorm",
        str(float(dic_raw.get("cutoff_diffnorm", 1.0e-6))),
        "--cutoff-iteration",
        str(int(dic_raw.get("cutoff_iteration", 50))),
        "--num-threads",
        str(int(dic_raw.get("num_threads", 1))),
        "--subset-truncation",
        "1" if bool(dic_raw.get("subset_truncation", False)) else "0",
        "--units-per-pixel",
        str(float(dic_raw.get("units_per_pixel", 1.0))),
        "--cutoff-corrcoef",
        str(float(dic_raw.get("cutoff_corrcoef", 0.6))),
        "--lenscoef",
        str(float(dic_raw.get("lenscoef", 0.0))),
    ]
    if os.name != "nt" or native_cli.suffix.lower() == ".exe":
        return args
    return ["wsl", "-e", *[_windows_path_to_wsl(arg) if _looks_like_windows_path(arg) else arg for arg in args]]


def _looks_like_windows_path(value: str) -> bool:
    return len(value) > 2 and value[1] == ":" and value[2] in ("\\", "/")


def _windows_path_to_wsl(value: str) -> str:
    path = Path(value).resolve()
    drive = path.drive.rstrip(":").lower()
    rest = path.as_posix().split(":/", 1)[1]
    return f"/mnt/{drive}/{rest}"


def _write_gray_binary(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    with path.open("wb") as handle:
        handle.write(np.asarray([width, height], dtype="<i4").tobytes())
        handle.write(np.asarray(image, dtype="<f8").ravel(order="F").tobytes())


def _write_mask_binary(path: Path, mask: np.ndarray) -> None:
    height, width = mask.shape[:2]
    with path.open("wb") as handle:
        handle.write(np.asarray([width, height], dtype="<i4").tobytes())
        handle.write(np.asarray(mask, dtype=np.uint8).ravel(order="F").tobytes())


def _read_result_binary(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"Native DIC2D result is truncated: {path}")
    dims = np.frombuffer(data[:8], dtype="<i4")
    width = int(dims[0])
    height = int(dims[1])
    count = width * height
    offset = 8
    bytes_double = count * np.dtype("<f8").itemsize
    expected_v2 = offset + bytes_double * 11 + count
    expected_v1 = offset + bytes_double * 3 + count
    if len(data) not in (expected_v2, expected_v1):
        raise ValueError(f"Native DIC2D result has unexpected byte size: {path}")

    def read_double_matrix() -> np.ndarray:
        nonlocal offset
        matrix = np.frombuffer(data[offset : offset + bytes_double], dtype="<f8").reshape((height, width), order="F").copy()
        offset += bytes_double
        return matrix

    if len(data) == expected_v2:
        x_ref = read_double_matrix()
        y_ref = read_double_matrix()
        x_def = read_double_matrix()
        y_def = read_double_matrix()
        u = read_double_matrix()
        v = read_double_matrix()
        ux = read_double_matrix()
        uy = read_double_matrix()
        vx = read_double_matrix()
        vy = read_double_matrix()
        corrcoef = read_double_matrix()
        schema_version = 2
    else:
        u = read_double_matrix()
        v = read_double_matrix()
        corrcoef = read_double_matrix()
        y_ref, x_ref = np.indices((height, width), dtype=np.float64)
        x_def = x_ref + u
        y_def = y_ref + v
        ux = np.zeros_like(u)
        uy = np.zeros_like(u)
        vx = np.zeros_like(u)
        vy = np.zeros_like(u)
        schema_version = 1
    valid = np.frombuffer(data[offset : offset + count], dtype=np.uint8).reshape((height, width), order="F").astype(bool)
    return {
        "output_schema_version": np.asarray(schema_version, dtype=np.int32),
        "x_ref": x_ref,
        "y_ref": y_ref,
        "x_def": x_def,
        "y_def": y_def,
        "u": u,
        "v": v,
        "ux": ux,
        "uy": uy,
        "vx": vx,
        "vy": vy,
        "corrcoef": corrcoef,
        "valid": valid,
        "reduced_width": np.asarray(width, dtype=np.int32),
        "reduced_height": np.asarray(height, dtype=np.int32),
    }


def _resolve_roi_path(config: MDICConfig, cam_name: str, auto_mask_dir: Path, roi_raw: dict[str, Any]) -> tuple[Path, str]:
    if str(roi_raw.get("user_roi_mode", "last_image")) == "last_image":
        user_roi = _last_extra_image_in_camera_folder(config, cam_name)
        if user_roi is not None:
            return user_roi, "user_last_image"
    auto_roi = _find_mask(auto_mask_dir, cam_name)
    if auto_roi is None:
        raise FileNotFoundError(f"No user ROI image and no automatic mask for {cam_name} in {auto_mask_dir}")
    return auto_roi, "auto_mask"


def _last_extra_image_in_camera_folder(config: MDICConfig, cam_name: str) -> Path | None:
    cam_dir = config.speckle_root / cam_name
    if not cam_dir.exists():
        return None
    images = sorted(
        [path for path in cam_dir.iterdir() if path.suffix.lower() in config.data.image_extensions],
        key=lambda path: _natural_key(path.name),
    )
    if not images:
        return None
    data_frames = {config.data.reference_frame, *config.data.deformed_frames}
    last = images[-1]
    return last if last.name not in data_frames else None


def _find_mask(mask_dir: Path, cam_name: str) -> Path | None:
    for ext in MASK_EXTENSIONS:
        for candidate in (
            mask_dir / f"{cam_name}_mask{ext}",
            mask_dir / f"{cam_name}{ext}",
            mask_dir / cam_name / f"mask{ext}",
            mask_dir / cam_name / f"{cam_name}_mask{ext}",
        ):
            if candidate.exists():
                return candidate
    matches = sorted(
        [path for path in mask_dir.glob(f"{cam_name}*") if path.suffix.lower() in MASK_EXTENSIONS],
        key=lambda path: _natural_key(path.name),
    )
    return matches[0] if matches else None


def _read_mask(path: Path, threshold: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.dtype == bool:
            return np.asarray(data, dtype=bool)
        if np.issubdtype(data.dtype, np.number) and float(np.nanmax(data)) <= 1.0:
            return np.asarray(data > 0, dtype=bool)
    else:
        import cv2

        data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise ValueError(f"Could not read ROI mask: {path}")
    if data.ndim == 3:
        import cv2

        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return np.asarray(data > threshold, dtype=bool)


def _select_seed(
    observations: dict[str, np.ndarray],
    cam_id: int,
    roi_mask: np.ndarray,
    dic_raw: dict[str, Any],
    reference_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    uv = observations["uv"][observations["cam_indices"] == cam_id].astype(np.float64)
    height, width = roi_mask.shape[:2]
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    subset_radius = int(dic_raw.get("subset_radius", 20))
    for point in uv:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height and bool(roi_mask[y, x]):
            seed = _snap_to_ncorr_grid_inside_roi(point, roi_mask, dic_raw)
            score = 0.0
            if reference_image is not None:
                sx, sy = int(seed[0]), int(seed[1])
                patch = reference_image[
                    sy - subset_radius : sy + subset_radius + 1,
                    sx - subset_radius : sx + subset_radius + 1,
                ]
                if patch.size:
                    score = float(np.std(patch))
            candidates.append((score, point, seed))
    if candidates:
        _, observation, seed = max(candidates, key=lambda item: item[0])
        return observation, seed
    raise ValueError("No COLMAP observation falls inside the ROI mask.")


def _snap_to_ncorr_grid_inside_roi(observation: np.ndarray, roi_mask: np.ndarray, dic_raw: dict[str, Any]) -> np.ndarray:
    height, width = roi_mask.shape[:2]
    spacing = int(dic_raw.get("subset_spacing", 0))
    radius_px = int(dic_raw.get("subset_radius", 10))
    subset_truncation = bool(dic_raw.get("subset_truncation", False))
    step = spacing + 1
    if step <= 0:
        raise ValueError("dic2d.subset_spacing must be non-negative")

    x0 = int(np.clip(round(float(observation[0]) / step) * step, 0, width - 1))
    y0 = int(np.clip(round(float(observation[1]) / step) * step, 0, height - 1))
    if _grid_point_allowed(roi_mask, x0, y0, radius_px, subset_truncation):
        return np.array([x0, y0], dtype=np.float64)

    max_radius = max(width, height) // step + 2
    for ring in range(1, max_radius + 1):
        best: tuple[float, int, int] | None = None
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                if max(abs(dx), abs(dy)) != ring:
                    continue
                x = x0 + dx * step
                y = y0 + dy * step
                if not _grid_point_allowed(roi_mask, x, y, radius_px, subset_truncation):
                    continue
                dist2 = (float(observation[0]) - x) ** 2 + (float(observation[1]) - y) ** 2
                if best is None or dist2 < best[0]:
                    best = (dist2, x, y)
        if best is not None:
            return np.array([best[1], best[2]], dtype=np.float64)
    raise ValueError("No Ncorr grid point near the COLMAP observation falls inside the ROI mask.")


def _grid_point_allowed(roi_mask: np.ndarray, x: int, y: int, radius_px: int, subset_truncation: bool) -> bool:
    height, width = roi_mask.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height or not bool(roi_mask[y, x]):
        return False
    if subset_truncation:
        return True
    return x >= radius_px and y >= radius_px and x < width - radius_px and y < height - radius_px


def _count_reference_grid_points(roi_mask: np.ndarray, dic_raw: dict[str, Any]) -> int:
    spacing = int(dic_raw.get("subset_spacing", 0))
    radius_px = int(dic_raw.get("subset_radius", 10))
    subset_truncation = bool(dic_raw.get("subset_truncation", False))
    step = spacing + 1
    if step <= 0:
        raise ValueError("dic2d.subset_spacing must be non-negative")
    height, width = roi_mask.shape[:2]
    count = 0
    for x in range(0, width, step):
        for y in range(0, height, step):
            if _grid_point_allowed(roi_mask, x, y, radius_px, subset_truncation):
                count += 1
    return count


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _natural_key(text: str) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _write_dic2d_report(config: MDICConfig, report: dict[str, Any]) -> None:
    logs_dir = config.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "dic2d_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
