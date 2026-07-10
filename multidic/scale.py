from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import MDICConfig


IMAGE_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


@dataclass
class ChessboardScaleConfig:
    data_dir: Path
    sfm_dir: Path
    image_dir: str
    output_dir: Path
    image_name: str | None = None
    inner_cols: int = 9
    inner_rows: int = 7
    square_size: float = 10.0
    square_size_unit: str = "mm"
    pair_selection: str = "all_pairs"
    scale_stat: str = "trimmed_mean"
    trim_fraction: float = 0.20
    max_pair_edge_cv: float = 0.08
    max_edge_cv_warning: float = 0.05
    corner_order_edge_cv_weight: float = 1.0
    subpix_window: int = 11
    min_common_corners: int = 12
    max_reprojection_error_px: float = 3.0
    save_overlays: bool = True


def run_scale(config: MDICConfig) -> dict[str, Any]:
    cfg = _scale_config(config)
    report: dict[str, Any] = {
        "ok": False,
        "project": config.project.name,
        "method": "chessboard",
        "sfm_dir": str(cfg.sfm_dir),
        "image_root": str(cfg.data_dir / cfg.image_dir),
        "output_dir": str(cfg.output_dir),
        "errors": [],
        "warnings": [],
    }
    try:
        outputs, payload = run_chessboard_scale(cfg)
    except Exception as exc:
        report["errors"].append(f"Scale failed: {type(exc).__name__}: {exc}")
        _write_scale_report(config, report)
        return report

    report["ok"] = True
    report["outputs"] = outputs
    report["scale"] = payload.get("scale", {})
    report["quality"] = payload.get("quality", {})
    report["selected_pair"] = payload.get("selected_pair", {})
    report["visible_cameras"] = payload.get("visible_cameras", [])
    report["warnings"] = payload.get("warnings", [])
    _write_scale_report(config, report)
    return report


def run_chessboard_scale(cfg: ChessboardScaleConfig) -> tuple[dict[str, str], dict[str, Any]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = cfg.output_dir / "detections"
    if cfg.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_sfm_cameras(cfg.sfm_dir)
    cam_names = [str(name) for name in cameras["cam_names"]]
    K = np.asarray(cameras["K"], dtype=np.float64)
    dist = np.asarray(cameras["dist"], dtype=np.float64)
    R = np.asarray(cameras["R"], dtype=np.float64)
    t = np.asarray(cameras["t"], dtype=np.float64).reshape(len(cam_names), 3)
    centers = np.asarray(cameras["camera_centers_world"], dtype=np.float64)

    pattern_size = (int(cfg.inner_cols), int(cfg.inner_rows))
    n_corners = cfg.inner_cols * cfg.inner_rows
    image_root = cfg.data_dir / cfg.image_dir
    detections: dict[int, np.ndarray] = {}
    detection_records: list[dict[str, Any]] = []
    image_paths: dict[int, str] = {}

    for cam_id, cam_name in enumerate(cam_names):
        cam_dir = image_root / cam_name
        image_path = find_named_image(cam_dir, cfg.image_name) if cfg.image_name else find_single_image(cam_dir)
        ok, corners, image = detect_chessboard_corners(image_path, pattern_size, cfg.subpix_window)
        detected = bool(ok and len(corners) == n_corners)
        if detected:
            detections[cam_id] = corners
        image_paths[cam_id] = str(image_path)
        detection_records.append(
            {
                "camera_id": cam_id,
                "camera_name": cam_name,
                "image_path": str(image_path),
                "detected": detected,
                "num_corners": int(len(corners)),
            }
        )
        if cfg.save_overlays:
            save_detection_overlay(overlay_dir / f"{cam_name}_corners.png", image, pattern_size, bool(ok), corners)

    visible = sorted(detections.keys())
    if len(visible) < 2:
        raise RuntimeError(f"Need at least two visible cameras, got {len(visible)}.")

    if cfg.pair_selection == "all_pairs":
        candidate_pairs = [(a, b) for pos, a in enumerate(visible) for b in visible[pos + 1 :]]
    else:
        candidate_pairs = [select_camera_pair(visible, cam_names, centers, cfg.pair_selection)]

    pair_errors: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for cam_a, cam_b in candidate_pairs:
        try:
            pair_results.append(evaluate_camera_pair(cam_a, cam_b, detections, cam_names, centers, K, dist, R, t, cfg))
        except RuntimeError as exc:
            pair_errors.append(
                {
                    "camera_ids": [int(cam_a), int(cam_b)],
                    "camera_names": [cam_names[cam_a], cam_names[cam_b]],
                    "error": str(exc),
                }
            )
    if not pair_results:
        raise RuntimeError(f"No valid chessboard camera pair. Pair errors: {pair_errors}")

    reliable_pairs = [
        pair
        for pair in pair_results
        if pair["quality"]["edge_cv"] <= float(cfg.max_pair_edge_cv)
        and pair["quality"]["num_valid_edges"] >= int(cfg.min_common_corners)
    ]
    warnings: list[str] = []
    if not reliable_pairs:
        reliable_pairs = sorted(pair_results, key=lambda p: p["quality"]["quality_score"])[:1]
        warnings.append(
            "No pair passed max_pair_edge_cv; falling back to best quality_score pair "
            f"({reliable_pairs[0]['camera_names']})."
        )

    selected_pair = min(reliable_pairs, key=lambda p: p["quality"]["quality_score"])
    if cfg.pair_selection == "all_pairs":
        sfm_to_world_scale = float(np.median([pair["scale"] for pair in reliable_pairs]))
        sfm_square_size = float(cfg.square_size / sfm_to_world_scale)
    else:
        sfm_to_world_scale = float(selected_pair["scale"])
        sfm_square_size = float(selected_pair["sfm_square_size"])

    world_to_sfm_scale = float(sfm_square_size / cfg.square_size)
    edge_lengths = selected_pair["edge_lengths_sfm"]
    edge_stats = selected_pair["edge_stats"]
    points = selected_pair["points"]
    valid = selected_pair["valid_corners"]
    cam_a, cam_b = selected_pair["camera_ids"]
    if selected_pair["quality"]["edge_cv"] > float(cfg.max_edge_cv_warning):
        warnings.append(
            "Selected chessboard pair has high edge_cv; scale may be unreliable. "
            f"edge_cv={selected_pair['quality']['edge_cv']:.6g}, "
            f"warning_threshold={cfg.max_edge_cv_warning:.6g}."
        )

    pair_records = [
        pair_record_for_json(pair, reliable=pair in reliable_pairs)
        for pair in sorted(pair_results, key=lambda p: p["quality"]["quality_score"])
    ]

    payload = {
        "config": _config_for_json(cfg),
        "sfm_dir": str(cfg.sfm_dir),
        "image_root": str(image_root),
        "visible_cameras": [cam_names[i] for i in visible],
        "selected_pair": {
            "camera_ids": [int(cam_a), int(cam_b)],
            "camera_names": [cam_names[cam_a], cam_names[cam_b]],
            "pair_selection": cfg.pair_selection,
            "corner_order_second_view": selected_pair["corner_order_second_view"],
            "baseline_sfm": float(np.linalg.norm(centers[cam_a] - centers[cam_b])),
        },
        "scale": {
            "scale_stat": str(cfg.scale_stat),
            "trim_fraction": float(cfg.trim_fraction),
            "sfm_square_size_mean": float(edge_stats["mean"]),
            "sfm_square_size_selected": sfm_square_size,
            "selected_pair_sfm_square_size_mean": float(edge_stats["mean"]),
            "sfm_square_size_median": float(edge_stats["median"]),
            "sfm_square_size_std": float(edge_stats["std"]),
            "physical_square_size": float(cfg.square_size),
            "physical_square_size_unit": cfg.square_size_unit,
            "sfm_to_world_scale": sfm_to_world_scale,
            "world_to_sfm_scale": world_to_sfm_scale,
            "unit": "physical_units_per_sfm_unit",
        },
        "quality": {
            "num_detected_cameras": int(len(visible)),
            "num_triangulated_corners": int(len(points)),
            "num_valid_corners": int(np.count_nonzero(valid)),
            "num_valid_edges": int(len(edge_lengths)),
            "reprojection_error_px_mean": selected_pair["quality"]["reprojection_error_px_mean"],
            "reprojection_error_px_median": selected_pair["quality"]["reprojection_error_px_median"],
            "reprojection_error_px_max": selected_pair["quality"]["reprojection_error_px_max"],
            "edge_cv": selected_pair["quality"]["edge_cv"],
            "selected_pair_quality_score": selected_pair["quality"]["quality_score"],
            "num_candidate_pairs": int(len(pair_results)),
            "num_reliable_pairs": int(len(reliable_pairs)),
            "max_pair_edge_cv": float(cfg.max_pair_edge_cv),
        },
        "pair_scale_records": pair_records,
        "pair_errors": pair_errors,
        "warnings": warnings,
        "detections": detection_records,
    }

    npz_path = cfg.output_dir / "chessboard_triangulation.npz"
    json_path = cfg.output_dir / "sfm2world_scale.json"
    np.savez_compressed(
        npz_path,
        points_sfm=points,
        corners_uv_a=selected_pair["corners_uv_a"],
        corners_uv_b=selected_pair["corners_uv_b"],
        valid_corners=valid,
        edge_lengths_sfm=edge_lengths,
        selected_camera_ids=np.asarray([cam_a, cam_b], dtype=np.int64),
        selected_camera_names=np.asarray([cam_names[cam_a], cam_names[cam_b]]),
        visible_camera_ids=np.asarray(visible, dtype=np.int64),
        visible_camera_names=np.asarray([cam_names[i] for i in visible]),
        reliable_pair_scales=np.asarray([pair["scale"] for pair in reliable_pairs], dtype=np.float64),
        candidate_pair_scales=np.asarray([pair["scale"] for pair in pair_results], dtype=np.float64),
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    outputs = {"scale_json": str(json_path), "triangulation_npz": str(npz_path)}
    if cfg.save_overlays:
        outputs["detection_dir"] = str(overlay_dir)
    return outputs, payload


def load_sfm_cameras(sfm_dir: Path) -> dict[str, np.ndarray]:
    path = sfm_dir / "cameras.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def find_named_image(cam_dir: Path, stem: str | None) -> Path:
    if not stem:
        return find_single_image(cam_dir)
    for ext in IMAGE_EXTENSIONS:
        path = cam_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image named {stem} with supported extension in {cam_dir}")


def find_single_image(cam_dir: Path) -> Path:
    if not cam_dir.is_dir():
        raise FileNotFoundError(cam_dir)
    images = sorted([path for path in cam_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: p.name)
    if len(images) != 1:
        raise FileNotFoundError(f"Expected exactly one calibration image in {cam_dir}, found {len(images)}")
    return images[0]


def detect_chessboard_corners(
    image_path: Path,
    pattern_size: tuple[int, int],
    subpix_window: int,
) -> tuple[bool, np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(image, pattern_size, flags)
    if not ok or corners is None:
        return False, np.zeros((0, 2), dtype=np.float64), image
    window = max(3, int(subpix_window) | 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    corners = cv2.cornerSubPix(image, corners, (window // 2, window // 2), (-1, -1), criteria)
    return True, corners.reshape(-1, 2).astype(np.float64), image


def save_detection_overlay(path: Path, image: np.ndarray, pattern_size: tuple[int, int], ok: bool, corners: np.ndarray) -> None:
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    draw_corners = corners.reshape(-1, 1, 2).astype(np.float32) if len(corners) else None
    cv2.drawChessboardCorners(vis, pattern_size, draw_corners, ok)
    cv2.imwrite(str(path), vis)


def select_camera_pair(visible_indices: list[int], cam_names: list[str], centers: np.ndarray, strategy: str) -> tuple[int, int]:
    if len(visible_indices) < 2:
        raise ValueError("Need at least two chessboard-visible cameras.")
    if strategy == "middle":
        order_numbers = [_camera_order_value(cam_names[i], i) for i in range(len(cam_names))]
        n_cameras = max(order_numbers) + 1 if order_numbers else len(cam_names)
        visible_order = [_camera_order_value(cam_names[i], i) for i in visible_indices]
        by_order = dict(zip(visible_order, visible_indices))
        arc = _continuous_visible_arc(visible_order, n_cameras)
        arc_indices = [by_order[i] for i in arc]
        mid = len(arc_indices) // 2
        if len(arc_indices) % 2 == 0:
            pair = (arc_indices[mid - 1], arc_indices[mid])
        else:
            pair = (arc_indices[max(0, mid - 1)], arc_indices[min(len(arc_indices) - 1, mid + 1)])
        return tuple(sorted(pair))
    if strategy == "max_baseline":
        best_pair: tuple[int, int] | None = None
        best_distance = -np.inf
        for pos, i in enumerate(visible_indices):
            for j in visible_indices[pos + 1 :]:
                distance = float(np.linalg.norm(centers[i] - centers[j]))
                if distance > best_distance:
                    best_pair = (i, j)
                    best_distance = distance
        if best_pair is None:
            raise ValueError("Could not select a visible camera pair.")
        return best_pair
    raise ValueError(f"Unknown pair_selection={strategy!r}; expected middle, max_baseline, or all_pairs.")


def evaluate_camera_pair(
    cam_a: int,
    cam_b: int,
    detections: dict[int, np.ndarray],
    cam_names: list[str],
    centers: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: ChessboardScaleConfig,
) -> dict[str, Any]:
    uv_a = detections[cam_a]
    uv_b = detections[cam_b]
    if len(uv_a) < cfg.min_common_corners or len(uv_b) < cfg.min_common_corners:
        raise RuntimeError(
            f"Pair {cam_names[cam_a]}-{cam_names[cam_b]} has too few common corners: "
            f"{len(uv_a)} and {len(uv_b)}; min_common_corners={cfg.min_common_corners}"
        )

    points, uv_b_aligned, corner_order_b, _ = triangulate_with_best_corner_order(
        uv_a, uv_b, cfg, K[cam_a], K[cam_b], dist[cam_a], dist[cam_b], R[cam_a], R[cam_b], t[cam_a], t[cam_b]
    )
    reproj_a = project_points(points, K[cam_a], dist[cam_a], R[cam_a], t[cam_a])
    reproj_b = project_points(points, K[cam_b], dist[cam_b], R[cam_b], t[cam_b])
    err_a = np.linalg.norm(reproj_a - uv_a, axis=1)
    err_b = np.linalg.norm(reproj_b - uv_b_aligned, axis=1)
    mean_reproj = 0.5 * (err_a + err_b)
    valid = mean_reproj <= float(cfg.max_reprojection_error_px)
    if np.count_nonzero(valid) < cfg.min_common_corners:
        raise RuntimeError(
            f"Pair {cam_names[cam_a]}-{cam_names[cam_b]} only has "
            f"{np.count_nonzero(valid)} valid corners after reprojection filtering."
        )

    horizontal, vertical = grid_edge_lengths(points, cfg.inner_rows, cfg.inner_cols)
    valid_grid = valid.reshape(cfg.inner_rows, cfg.inner_cols)
    valid_h = (valid_grid[:, 1:] & valid_grid[:, :-1]).reshape(-1)
    valid_v = (valid_grid[1:, :] & valid_grid[:-1, :]).reshape(-1)
    edge_lengths = np.concatenate([horizontal[valid_h], vertical[valid_v]])
    if len(edge_lengths) == 0:
        raise RuntimeError(f"Pair {cam_names[cam_a]}-{cam_names[cam_b]} has no valid chessboard edges.")

    edge_stats = edge_length_stats(edge_lengths, cfg.scale_stat, cfg.trim_fraction)
    sfm_square_size = float(edge_stats["selected"])
    sfm_to_world_scale = float(cfg.square_size / sfm_square_size)
    edge_cv = float(edge_stats["cv_mean"])
    baseline = float(np.linalg.norm(centers[cam_a] - centers[cam_b]))
    quality_score = edge_cv + 0.01 * float(np.median(mean_reproj)) + 1.0 / max(baseline, 1e-12)
    return {
        "camera_ids": [int(cam_a), int(cam_b)],
        "camera_names": [cam_names[cam_a], cam_names[cam_b]],
        "corner_order_second_view": corner_order_b,
        "baseline_sfm": baseline,
        "points": points,
        "corners_uv_a": uv_a,
        "corners_uv_b": uv_b_aligned,
        "valid_corners": valid,
        "edge_lengths_sfm": edge_lengths,
        "scale": sfm_to_world_scale,
        "sfm_square_size": sfm_square_size,
        "edge_stats": edge_stats,
        "quality": {
            "num_triangulated_corners": int(len(points)),
            "num_valid_corners": int(np.count_nonzero(valid)),
            "num_valid_edges": int(len(edge_lengths)),
            "reprojection_error_px_mean": float(np.mean(mean_reproj)),
            "reprojection_error_px_median": float(np.median(mean_reproj)),
            "reprojection_error_px_max": float(np.max(mean_reproj)),
            "edge_cv": edge_cv,
            "quality_score": float(quality_score),
        },
    }


def triangulate_with_best_corner_order(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    cfg: ChessboardScaleConfig,
    K_a: np.ndarray,
    K_b: np.ndarray,
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    R_a: np.ndarray,
    R_b: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    best: tuple[float, np.ndarray, np.ndarray, str, np.ndarray] | None = None
    for name, candidate_b in corner_order_candidates(uv_b, cfg.inner_rows, cfg.inner_cols):
        points = triangulate_points(uv_a, candidate_b, K_a, K_b, dist_a, dist_b, R_a, R_b, t_a, t_b)
        reproj_a = project_points(points, K_a, dist_a, R_a, t_a)
        reproj_b = project_points(points, K_b, dist_b, R_b, t_b)
        err = 0.5 * (np.linalg.norm(reproj_a - uv_a, axis=1) + np.linalg.norm(reproj_b - candidate_b, axis=1))
        horizontal, vertical = grid_edge_lengths(points, cfg.inner_rows, cfg.inner_cols)
        candidate_edges = np.concatenate([horizontal, vertical])
        edge_cv = float(np.std(candidate_edges) / max(abs(np.mean(candidate_edges)), 1e-12))
        score = float(np.median(err)) + float(cfg.corner_order_edge_cv_weight) * edge_cv
        if best is None or score < best[0]:
            best = (score, points, candidate_b, name, err)
    if best is None:
        raise RuntimeError("Failed to evaluate chessboard corner order candidates.")
    _, points, aligned_b, order_name, mean_err = best
    return points, aligned_b, order_name, mean_err


def corner_order_candidates(corners: np.ndarray, inner_rows: int, inner_cols: int) -> list[tuple[str, np.ndarray]]:
    grid = corners.reshape(inner_rows, inner_cols, 2)
    return [
        ("as_detected", grid.reshape(-1, 2)),
        ("reverse_all", grid[::-1, ::-1].reshape(-1, 2)),
        ("flip_rows", grid[::-1, :].reshape(-1, 2)),
        ("flip_cols", grid[:, ::-1].reshape(-1, 2)),
    ]


def triangulate_points(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    K_a: np.ndarray,
    K_b: np.ndarray,
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    R_a: np.ndarray,
    R_b: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
) -> np.ndarray:
    uv_a = undistort_to_pixel(uv_a, K_a, dist_a)
    uv_b = undistort_to_pixel(uv_b, K_b, dist_b)
    P_a = K_a @ np.concatenate([R_a, t_a.reshape(3, 1)], axis=1)
    P_b = K_b @ np.concatenate([R_b, t_b.reshape(3, 1)], axis=1)
    homog = cv2.triangulatePoints(P_a, P_b, uv_a.T, uv_b.T)
    xyz = (homog[:3] / homog[3:4]).T
    return xyz.astype(np.float64)


def undistort_to_pixel(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if np.all(np.abs(dist) < 1e-12):
        return uv.astype(np.float64)
    undist = cv2.undistortPoints(
        uv.reshape(-1, 1, 2).astype(np.float64),
        K.astype(np.float64),
        dist.astype(np.float64),
        P=K.astype(np.float64),
    )
    return undist.reshape(-1, 2)


def project_points(points: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    p_cam = R @ points.T + t.reshape(3, 1)
    x = p_cam[0] / p_cam[2]
    y = p_cam[1] / p_cam[2]
    k1, k2 = float(dist[0]), float(dist[1])
    if abs(k1) > 1e-12 or abs(k2) > 1e-12:
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x = x * radial
        y = y * radial
    return np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1)


def grid_edge_lengths(points: np.ndarray, inner_rows: int, inner_cols: int) -> tuple[np.ndarray, np.ndarray]:
    grid = points.reshape(inner_rows, inner_cols, 3)
    horizontal = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).reshape(-1)
    vertical = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).reshape(-1)
    return horizontal, vertical


def edge_length_stats(edge_lengths: np.ndarray, scale_stat: str, trim_fraction: float) -> dict[str, float | int]:
    edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
    selected = robust_edge_size(edge_lengths, scale_stat, trim_fraction)
    mean = float(np.mean(edge_lengths))
    median = float(np.median(edge_lengths))
    std = float(np.std(edge_lengths))
    return {
        "count": int(len(edge_lengths)),
        "selected": selected,
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(np.min(edge_lengths)),
        "p05": float(np.percentile(edge_lengths, 5)),
        "p25": float(np.percentile(edge_lengths, 25)),
        "p75": float(np.percentile(edge_lengths, 75)),
        "p90": float(np.percentile(edge_lengths, 90)),
        "p95": float(np.percentile(edge_lengths, 95)),
        "max": float(np.max(edge_lengths)),
        "cv_mean": float(std / max(abs(mean), 1e-12)),
        "cv_selected": float(std / max(abs(selected), 1e-12)),
    }


def robust_edge_size(edge_lengths: np.ndarray, scale_stat: str, trim_fraction: float) -> float:
    edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
    edge_lengths = edge_lengths[np.isfinite(edge_lengths)]
    if len(edge_lengths) == 0:
        raise RuntimeError("Cannot compute chessboard scale from no valid edges.")
    stat = str(scale_stat).lower()
    if stat == "mean":
        return float(np.mean(edge_lengths))
    if stat == "median":
        return float(np.median(edge_lengths))
    if stat == "p25":
        return float(np.percentile(edge_lengths, 25))
    if stat == "trimmed_mean":
        trim = min(max(float(trim_fraction), 0.0), 0.45)
        ordered = np.sort(edge_lengths)
        lo = int(np.floor(len(ordered) * trim))
        hi = int(np.ceil(len(ordered) * (1.0 - trim)))
        trimmed = ordered[lo:hi]
        return float(np.mean(trimmed if len(trimmed) else ordered))
    raise ValueError(f"Unknown scale_stat={scale_stat!r}; expected mean, median, trimmed_mean, or p25.")


def pair_record_for_json(pair: dict[str, Any], *, reliable: bool) -> dict[str, Any]:
    return {
        "camera_ids": pair["camera_ids"],
        "camera_names": pair["camera_names"],
        "corner_order_second_view": pair["corner_order_second_view"],
        "baseline_sfm": pair["baseline_sfm"],
        "sfm_square_size": pair["sfm_square_size"],
        "sfm_to_world_scale": pair["scale"],
        "edge_stats": pair["edge_stats"],
        "quality": pair["quality"],
        "reliable": bool(reliable),
    }


def _camera_order_value(name: str, fallback: int) -> int:
    if "_" in name:
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return fallback


def _continuous_visible_arc(indices: list[int], n_cameras: int) -> list[int]:
    if not indices:
        return []
    ordered = sorted(indices)
    if len(ordered) <= 2:
        return ordered
    gaps = []
    for idx, value in enumerate(ordered):
        nxt = ordered[(idx + 1) % len(ordered)]
        gaps.append((nxt - value) % n_cameras)
    break_after = int(np.argmax(gaps))
    return ordered[break_after + 1 :] + ordered[: break_after + 1]


def _scale_config(config: MDICConfig) -> ChessboardScaleConfig:
    scale_raw = config.raw.get("sfm2world")
    if not isinstance(scale_raw, dict):
        scale_raw = config.raw.get("scale_correction", {})
    if not isinstance(scale_raw, dict):
        scale_raw = {}
    meta = _load_checkerboard_meta(config, scale_raw)
    board = meta.get("board", {}) if isinstance(meta, dict) else {}

    inner_cols = int(scale_raw.get("inner_cols", board.get("inner_corners_cols", 9)))
    inner_rows = int(scale_raw.get("inner_rows", board.get("inner_corners_rows", 7)))
    square_size = float(scale_raw.get("square_size", board.get("square_size_mm", 10.0)))
    output_dir = config.result_root / str(scale_raw.get("output_dir", "scale"))
    colmap_cfg = config.raw.get("colmap", {})
    workspace = str(colmap_cfg.get("workspace", "colmap")) if isinstance(colmap_cfg, dict) else "colmap"
    return ChessboardScaleConfig(
        data_dir=config.case_root,
        sfm_dir=config.result_root / "sfm" / workspace,
        image_dir=str(scale_raw.get("image_dir", config.data.calibration_dir)),
        image_name=scale_raw.get("image_name"),
        output_dir=output_dir,
        inner_cols=inner_cols,
        inner_rows=inner_rows,
        square_size=square_size,
        square_size_unit=str(scale_raw.get("square_size_unit", config.raw.get("scale_correction", {}).get("square_size_unit", "mm"))),
        pair_selection=str(scale_raw.get("pair_selection", "all_pairs")),
        scale_stat=str(scale_raw.get("scale_stat", "trimmed_mean")),
        trim_fraction=float(scale_raw.get("trim_fraction", 0.20)),
        max_pair_edge_cv=float(scale_raw.get("max_pair_edge_cv", 0.08)),
        max_edge_cv_warning=float(scale_raw.get("max_edge_cv_warning", 0.05)),
        corner_order_edge_cv_weight=float(scale_raw.get("corner_order_edge_cv_weight", 1.0)),
        subpix_window=int(scale_raw.get("subpix_window", 11)),
        min_common_corners=int(scale_raw.get("min_common_corners", 12)),
        max_reprojection_error_px=float(scale_raw.get("max_reprojection_error_px", 3.0)),
        save_overlays=bool(scale_raw.get("save_overlays", True)),
    )


def _load_checkerboard_meta(config: MDICConfig, scale_raw: dict[str, Any]) -> dict[str, Any]:
    meta_path_text = scale_raw.get("checkerboard_meta")
    if not meta_path_text:
        return {}
    path = Path(str(meta_path_text))
    resolved = path if path.is_absolute() else config.case_root / path
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _config_for_json(cfg: ChessboardScaleConfig) -> dict[str, Any]:
    data = asdict(cfg)
    for key in ("data_dir", "sfm_dir", "output_dir"):
        data[key] = str(data[key])
    return data


def _write_scale_report(config: MDICConfig, report: dict[str, Any]) -> None:
    logs_dir = config.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "scale_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
