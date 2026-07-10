from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .config import MDICConfig


MASK_EXTENSIONS = (".npy", ".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass
class ROIConfig:
    use_external: bool = False
    external_roi_dir: str | None = None
    external_threshold: int = 127
    outlier_k: int = 6
    outlier_knn_scale: float = 4.0
    component_radius_scale: float = 8.0
    edge_scale: float = 8.0
    radius_scale: float = 6.0
    min_hole_area: int = 500
    tiny_hole_fill_area: int = 3000
    speckle_std_ratio: float = 0.35
    speckle_lap_ratio: float = 0.35
    speckle_grad_ratio: float = 0.35
    min_speckle_std: float = 6.0
    min_speckle_lap: float = 3.0
    overlay_alpha: float = 0.45
    dpi: int = 180


@dataclass
class MaskStepConfig:
    data_dir: Path
    sfm_dir: Path
    output_dir: Path
    user_mask_dir: Path | None
    use_user_mask_if_present: bool
    roi: ROIConfig


@dataclass
class CameraMask:
    cam_id: int
    cam_name: str
    mask: np.ndarray
    hull_mask: np.ndarray
    supported_mask: np.ndarray
    rejected_hole_mask: np.ndarray
    hull: np.ndarray
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    n_observations: int = 0
    n_points_after_outlier_filter: int = 0
    n_triangles_raw: int = 0
    n_triangles_valid: int = 0
    n_holes_detected: int = 0
    n_holes_filled_as_speckle: int = 0
    n_holes_rejected: int = 0
    reference_texture: dict[str, float] | None = None


def run_mask(config: MDICConfig) -> dict[str, Any]:
    cfg = _mask_config(config)
    report: dict[str, Any] = {
        "ok": False,
        "project": config.project.name,
        "sfm_dir": str(cfg.sfm_dir),
        "output_dir": str(cfg.output_dir),
        "user_mask_dir": str(cfg.user_mask_dir) if cfg.user_mask_dir else None,
        "errors": [],
        "warnings": [],
    }
    try:
        masks, outputs, payload = run_roi_masks(cfg)
    except Exception as exc:
        report["errors"].append(f"Mask generation failed: {type(exc).__name__}: {exc}")
        _write_mask_report(config, report)
        return report

    report["ok"] = True
    report["mode"] = payload.get("mode")
    report["camera_count"] = len(masks)
    report["outputs"] = outputs
    report["cameras"] = payload.get("cameras", [])
    report["warnings"] = payload.get("warnings", [])
    _write_mask_report(config, report)
    return report


def run_roi_masks(cfg: MaskStepConfig) -> tuple[list[CameraMask], dict[str, str], dict[str, Any]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = _load_npz(cfg.sfm_dir / "cameras.npz")
    observations = _load_npz(cfg.sfm_dir / "observations.npz")
    cam_names = [str(name) for name in cameras["cam_names"]]
    ref_images = _load_reference_images(cameras["image_paths"], cfg.data_dir, cam_names)

    user_dir = _complete_external_mask_dir(cam_names, cfg.user_mask_dir) if cfg.use_user_mask_if_present else None
    roi_cfg = cfg.roi
    if user_dir is not None:
        roi_cfg = ROIConfig(**{**asdict(cfg.roi), "use_external": True, "external_roi_dir": str(user_dir)})
        masks = _load_external_roi_masks(cam_names, ref_images, roi_cfg)
    else:
        masks = build_roi_masks_from_observations(cam_names, ref_images, observations, roi_cfg)

    payload = _save_masks(masks, roi_cfg, cfg.output_dir, ref_images)
    warnings: list[str] = []
    if cfg.user_mask_dir and user_dir is None and cfg.user_mask_dir.exists():
        warnings.append(f"User mask directory exists but is incomplete: {cfg.user_mask_dir}; automatic masks were generated.")
    payload["warnings"] = warnings
    outputs = {
        "mask_dir": str(cfg.output_dir / "mask"),
        "overlay_dir": str(cfg.output_dir / "overlay"),
        "debug_dir": str(cfg.output_dir / "debug"),
        "meta_json": str(cfg.output_dir / "auto_roi_meta.json"),
        "summary_png": str(cfg.output_dir / "auto_roi_summary.png"),
    }
    return masks, outputs, payload


def build_roi_masks_from_observations(
    cam_names: list[str],
    ref_images: list[np.ndarray],
    observations: dict[str, np.ndarray],
    config: ROIConfig,
) -> list[CameraMask]:
    masks: list[CameraMask] = []
    t0 = time.time()
    for cam_id, cam_name in enumerate(cam_names):
        image = ref_images[cam_id]
        uv = observations["uv"][observations["cam_indices"] == cam_id].astype(np.float64)
        masks.append(_build_single_auto_roi(cam_id, cam_name, image, uv, config))
    _ = t0
    return masks


def _build_single_auto_roi(cam_id: int, cam_name: str, image: np.ndarray, uv: np.ndarray, cfg: ROIConfig) -> CameraMask:
    height, width = image.shape[:2]
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv_in = uv[in_frame]
    uv_clean = _remove_feature_outliers(uv_in, cfg)

    if len(uv_clean) < 3:
        empty = np.zeros((height, width), dtype=bool)
        return CameraMask(
            cam_id=cam_id,
            cam_name=cam_name,
            mask=empty,
            hull_mask=empty.copy(),
            supported_mask=empty.copy(),
            rejected_hole_mask=empty.copy(),
            hull=np.empty((0, 2), dtype=np.float32),
            u_min=0.0,
            u_max=0.0,
            v_min=0.0,
            v_max=0.0,
            n_observations=int(len(uv_in)),
        )

    hull = cv2.convexHull(uv_clean.astype(np.float32)).reshape(-1, 2)
    hull_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [np.round(hull).astype(np.int32)], 1)
    hull_bool = hull_mask.astype(bool)

    supported_mask, n_tri_raw, n_tri_valid = _build_delaunay_support_mask(uv_clean, height, width, cfg)
    supported_mask &= hull_bool
    final_mask, rejected_hole_mask, hole_stats, ref_texture = _classify_and_fill_holes(image, hull_bool, supported_mask, cfg)
    u_min, u_max, v_min, v_max = _compute_bounds(final_mask)

    return CameraMask(
        cam_id=cam_id,
        cam_name=cam_name,
        mask=final_mask,
        hull_mask=hull_bool,
        supported_mask=supported_mask,
        rejected_hole_mask=rejected_hole_mask,
        hull=hull.astype(np.float32),
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        n_observations=int(len(uv_in)),
        n_points_after_outlier_filter=int(len(uv_clean)),
        n_triangles_raw=int(n_tri_raw),
        n_triangles_valid=int(n_tri_valid),
        n_holes_detected=int(hole_stats["detected"]),
        n_holes_filled_as_speckle=int(hole_stats["filled"]),
        n_holes_rejected=int(hole_stats["rejected"]),
        reference_texture=ref_texture,
    )


def _remove_feature_outliers(uv: np.ndarray, cfg: ROIConfig) -> np.ndarray:
    if len(uv) <= max(3, cfg.outlier_k + 1):
        return uv
    tree = cKDTree(uv)
    dists, _ = tree.query(uv, k=cfg.outlier_k + 1)
    nn = dists[:, 1]
    kth = dists[:, -1]
    median_nn = float(np.median(nn[nn > 0])) if np.any(nn > 0) else float(np.median(kth))
    if not np.isfinite(median_nn) or median_nn <= 0:
        return uv
    uv_density = uv[kth <= cfg.outlier_knn_scale * median_nn]
    if len(uv_density) < 3:
        uv_density = uv
    return _keep_largest_radius_component(uv_density, cfg.component_radius_scale * median_nn)


def _keep_largest_radius_component(uv: np.ndarray, radius: float) -> np.ndarray:
    if len(uv) < 3 or radius <= 0:
        return uv
    tree = cKDTree(uv)
    pairs = list(tree.query_pairs(radius))
    if not pairs:
        return uv
    parent = np.arange(len(uv))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)
    roots = np.array([find(i) for i in range(len(uv))])
    labels, counts = np.unique(roots, return_counts=True)
    keep = roots == labels[np.argmax(counts)]
    return uv[keep] if keep.sum() >= 3 else uv


def _build_delaunay_support_mask(uv: np.ndarray, height: int, width: int, cfg: ROIConfig) -> tuple[np.ndarray, int, int]:
    d_nn = _median_nn_distance(uv)
    if d_nn <= 0:
        return np.zeros((height, width), dtype=bool), 0, 0
    try:
        tri = Delaunay(uv)
    except Exception:
        return np.zeros((height, width), dtype=bool), 0, 0
    valid = _filter_triangles(uv, tri.simplices, d_nn, cfg)
    valid_tris = tri.simplices[valid]
    if len(valid_tris) == 0:
        return np.zeros((height, width), dtype=bool), int(len(tri.simplices)), 0
    return _rasterize_triangles(uv, valid_tris, height, width), int(len(tri.simplices)), int(len(valid_tris))


def _median_nn_distance(uv: np.ndarray) -> float:
    if len(uv) < 2:
        return 0.0
    dists, _ = cKDTree(uv).query(uv, k=2)
    positive = dists[:, 1][dists[:, 1] > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def _filter_triangles(uv: np.ndarray, tri_indices: np.ndarray, d_nn: float, cfg: ROIConfig) -> np.ndarray:
    vertices = uv[tri_indices]
    e0 = vertices[:, 1] - vertices[:, 0]
    e1 = vertices[:, 2] - vertices[:, 1]
    e2 = vertices[:, 0] - vertices[:, 2]
    l0 = np.linalg.norm(e0, axis=-1)
    l1 = np.linalg.norm(e1, axis=-1)
    l2 = np.linalg.norm(e2, axis=-1)
    l_max = np.maximum(np.maximum(l0, l1), l2)
    cross = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    area = 0.5 * np.abs(cross)
    radius = (l0 * l1 * l2) / (4.0 * np.maximum(area, 1e-12))
    return (l_max < cfg.edge_scale * d_nn) & (radius < cfg.radius_scale * d_nn)


def _rasterize_triangles(uv: np.ndarray, tri_indices: np.ndarray, height: int, width: int) -> np.ndarray:
    vertices = uv[tri_indices]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices[:, :, None, :].astype(np.int32), color=1)
    return mask.astype(bool)


def _classify_and_fill_holes(
    image: np.ndarray,
    hull_mask: np.ndarray,
    supported_mask: np.ndarray,
    cfg: ROIConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, float]]:
    candidate_holes = hull_mask & ~supported_mask
    final = supported_mask.copy()
    rejected = np.zeros_like(hull_mask, dtype=bool)
    ref_texture = _texture_metrics(image, supported_mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_holes.astype(np.uint8), connectivity=8)
    counts = {"detected": 0, "filled": 0, "rejected": 0}
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.min_hole_area:
            continue
        counts["detected"] += 1
        hole = labels == label
        if area <= cfg.tiny_hole_fill_area or _is_speckle_like(_texture_metrics(image, hole), ref_texture, cfg):
            final[hole] = True
            counts["filled"] += 1
        else:
            rejected[hole] = True
            counts["rejected"] += 1
    final &= hull_mask
    return final, rejected, counts, ref_texture


def _texture_metrics(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    if mask.sum() == 0:
        return {"std": 0.0, "lap_std": 0.0, "grad_mean": 0.0}
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    values = gray[mask]
    return {"std": float(np.std(values)), "lap_std": float(np.std(lap[mask])), "grad_mean": float(np.mean(grad[mask]))}


def _is_speckle_like(hole: dict[str, float], ref: dict[str, float], cfg: ROIConfig) -> bool:
    std_ok = hole["std"] >= max(cfg.min_speckle_std, cfg.speckle_std_ratio * ref["std"])
    lap_ok = hole["lap_std"] >= max(cfg.min_speckle_lap, cfg.speckle_lap_ratio * ref["lap_std"])
    grad_ok = hole["grad_mean"] >= cfg.speckle_grad_ratio * ref["grad_mean"]
    return bool(std_ok and lap_ok and grad_ok)


def _complete_external_mask_dir(cam_names: list[str], user_mask_dir: Path | None) -> Path | None:
    if user_mask_dir is None or not user_mask_dir.exists():
        return None
    for cam_name in cam_names:
        if _find_external_mask(user_mask_dir, cam_name) is None:
            return None
    return user_mask_dir


def _load_external_roi_masks(cam_names: list[str], ref_images: list[np.ndarray], cfg: ROIConfig) -> list[CameraMask]:
    if not cfg.external_roi_dir:
        raise ValueError("external_roi_dir must be set when use_external=True")
    roi_dir = Path(cfg.external_roi_dir)
    masks: list[CameraMask] = []
    for cam_id, cam_name in enumerate(cam_names):
        path = _find_external_mask(roi_dir, cam_name)
        if path is None:
            raise FileNotFoundError(f"No external ROI mask for {cam_name} in {roi_dir}")
        mask = _read_mask(path, cfg.external_threshold)
        expected_shape = ref_images[cam_id].shape[:2]
        if mask.shape != expected_shape:
            raise ValueError(f"Mask shape for {cam_name} is {mask.shape}, expected {expected_shape}: {path}")
        u_min, u_max, v_min, v_max = _compute_bounds(mask)
        masks.append(
            CameraMask(
                cam_id=cam_id,
                cam_name=cam_name,
                mask=mask,
                hull_mask=mask.copy(),
                supported_mask=mask.copy(),
                rejected_hole_mask=np.zeros_like(mask, dtype=bool),
                hull=np.empty((0, 2), dtype=np.float32),
                u_min=u_min,
                u_max=u_max,
                v_min=v_min,
                v_max=v_max,
            )
        )
    return masks


def _find_external_mask(mask_dir: Path, cam_name: str) -> Path | None:
    candidates: list[Path] = []
    for ext in MASK_EXTENSIONS:
        candidates.extend(
            [
                mask_dir / f"{cam_name}_mask{ext}",
                mask_dir / f"{cam_name}{ext}",
                mask_dir / cam_name / f"mask{ext}",
                mask_dir / cam_name / f"{cam_name}_mask{ext}",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(
        [path for path in mask_dir.glob(f"{cam_name}*") if path.suffix.lower() in MASK_EXTENSIONS],
        key=lambda p: _natural_key(p.name),
    )
    return matches[0] if matches else None


def _read_mask(path: Path, threshold: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        return data.astype(bool)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > int(threshold)


def _save_masks(masks: list[CameraMask], cfg: ROIConfig, output_dir: Path, ref_images: list[np.ndarray]) -> dict[str, Any]:
    mask_dir = output_dir / "mask"
    overlay_dir = output_dir / "overlay"
    debug_dir = output_dir / "debug"
    for path in (mask_dir, overlay_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"mode": "external" if cfg.use_external else "auto", "config": asdict(cfg), "cameras": []}
    for cm, ref in zip(masks, ref_images):
        np.save(mask_dir / f"{cm.cam_name}_mask.npy", cm.mask)
        cv2.imwrite(str(mask_dir / f"{cm.cam_name}_mask.png"), cm.mask.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / f"{cm.cam_name}_hull.png"), cm.hull_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / f"{cm.cam_name}_delaunay_supported.png"), cm.supported_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / f"{cm.cam_name}_rejected_holes.png"), cm.rejected_hole_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(overlay_dir / f"{cm.cam_name}_overlay.png"), _make_overlay(ref, cm, cfg))
        payload["cameras"].append(
            {
                "cam_id": cm.cam_id,
                "cam_name": cm.cam_name,
                "mask_pixels": int(cm.mask.sum()),
                "hull_pixels": int(cm.hull_mask.sum()),
                "supported_pixels": int(cm.supported_mask.sum()),
                "rejected_hole_pixels": int(cm.rejected_hole_mask.sum()),
                "u_min": cm.u_min,
                "u_max": cm.u_max,
                "v_min": cm.v_min,
                "v_max": cm.v_max,
                "n_observations": cm.n_observations,
                "n_points_after_outlier_filter": cm.n_points_after_outlier_filter,
                "n_triangles_raw": cm.n_triangles_raw,
                "n_triangles_valid": cm.n_triangles_valid,
                "n_holes_detected": cm.n_holes_detected,
                "n_holes_filled_as_speckle": cm.n_holes_filled_as_speckle,
                "n_holes_rejected": cm.n_holes_rejected,
                "reference_texture": cm.reference_texture,
            }
        )

    with (output_dir / "auto_roi_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _save_summary_grid(output_dir / "auto_roi_summary.png", masks, ref_images, cfg)
    return payload


def _make_overlay(image: np.ndarray, cm: CameraMask, cfg: ROIConfig) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    color = np.zeros_like(base)
    color[cm.mask] = (0, 180, 0)
    color[cm.rejected_hole_mask] = (0, 0, 220)
    overlay = cv2.addWeighted(base, 1.0 - cfg.overlay_alpha, color, cfg.overlay_alpha, 0)
    if len(cm.hull) >= 3:
        cv2.polylines(overlay, [np.round(cm.hull).astype(np.int32)], True, (255, 255, 255), 2)
    return overlay


def _save_summary_grid(path: Path, masks: list[CameraMask], ref_images: list[np.ndarray], cfg: ROIConfig) -> None:
    import matplotlib.pyplot as plt

    n = len(masks)
    cols = min(4, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), dpi=int(cfg.dpi), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle("ROI masks: green=ROI, red=rejected holes, white=max feature hull")
    for ax, cm, ref in zip(axes_arr, masks, ref_images):
        overlay = _make_overlay(ref, cm, cfg)
        ax.imshow(overlay[..., ::-1])
        ax.set_title(f"{cm.cam_name}  {cm.mask.sum() // 1000}K px")
        ax.set_axis_off()
    for ax in axes_arr[len(masks) :]:
        ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def _compute_bounds(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(cols.min()), float(cols.max()), float(rows.min()), float(rows.max())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_reference_images(image_paths: np.ndarray, data_dir: Path, cam_names: list[str]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for raw_path, cam_name in zip(image_paths, cam_names):
        path = Path(str(raw_path))
        if not path.exists():
            path = data_dir / "images" / cam_name / path.name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    return images


def _mask_config(config: MDICConfig) -> MaskStepConfig:
    raw = config.raw.get("mask", {})
    if not isinstance(raw, dict):
        raw = {}
    colmap_cfg = config.raw.get("colmap", {})
    workspace = str(colmap_cfg.get("workspace", "colmap")) if isinstance(colmap_cfg, dict) else "colmap"
    output_dir = config.result_root / str(raw.get("output_dir", "masks"))
    user_mask_dir_text = raw.get("user_mask_dir", "masks")
    user_mask_dir = _resolve_case_path(config.case_root, user_mask_dir_text) if user_mask_dir_text else None
    roi = ROIConfig(
        external_threshold=int(raw.get("external_threshold", 127)),
        outlier_k=int(raw.get("outlier_k", 6)),
        outlier_knn_scale=float(raw.get("outlier_knn_scale", 4.0)),
        component_radius_scale=float(raw.get("component_radius_scale", 8.0)),
        edge_scale=float(raw.get("edge_scale", 8.0)),
        radius_scale=float(raw.get("radius_scale", 6.0)),
        min_hole_area=int(raw.get("min_hole_area", 500)),
        tiny_hole_fill_area=int(raw.get("tiny_hole_fill_area", 3000)),
        speckle_std_ratio=float(raw.get("speckle_std_ratio", 0.35)),
        speckle_lap_ratio=float(raw.get("speckle_lap_ratio", 0.35)),
        speckle_grad_ratio=float(raw.get("speckle_grad_ratio", 0.35)),
        min_speckle_std=float(raw.get("min_speckle_std", 6.0)),
        min_speckle_lap=float(raw.get("min_speckle_lap", 3.0)),
        overlay_alpha=float(raw.get("overlay_alpha", 0.45)),
        dpi=int(raw.get("dpi", 180)),
    )
    return MaskStepConfig(
        data_dir=config.case_root,
        sfm_dir=config.result_root / "sfm" / workspace,
        output_dir=output_dir,
        user_mask_dir=user_mask_dir,
        use_user_mask_if_present=bool(raw.get("use_user_mask_if_present", True)),
        roi=roi,
    )


def _resolve_case_path(case_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else case_root / path


def _write_mask_report(config: MDICConfig, report: dict[str, Any]) -> None:
    logs_dir = config.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "mask_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]
