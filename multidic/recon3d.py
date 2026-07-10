from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import MDICConfig


def run_recon3d(config: MDICConfig) -> dict[str, Any]:
    cfg = _recon_config(config)
    report: dict[str, Any] = {
        "ok": False,
        "project": config.project.name,
        "method": "sfm_track_anchored",
        "backend": "numpy",
        "sfm_dir": str(cfg["sfm_dir"]),
        "dic2d_dir": str(cfg["dic2d_dir"]),
        "output_dir": str(cfg["output_dir"]),
        "frames": [],
        "errors": [],
        "warnings": [],
    }
    try:
        cameras = _load_npz(cfg["sfm_dir"] / "cameras.npz")
        observations = _load_npz(cfg["sfm_dir"] / "observations.npz")
        scale = _load_sfm_to_world_scale(config, cfg)
        cfg["output_dir"].mkdir(parents=True, exist_ok=True)
        backend = _native_backend()
        if backend is not None:
            report["backend"] = "native_recon3d"
        elif bool(cfg["prefer_native"]):
            report["warnings"].append("native_recon3d is unavailable; using NumPy fallback.")
        for frame_name in config.data.deformed_frames:
            frame_report = _run_recon3d_frame(config, cfg, cameras, observations, frame_name, scale, backend)
            report["frames"].append(frame_report)
            if not frame_report.get("ok"):
                report["errors"].append(f"{frame_name}: {frame_report.get('error', 'reconstruction failed')}")
    except Exception as exc:
        report["errors"].append(f"Recon3D failed: {type(exc).__name__}: {exc}")

    report["ok"] = not report["errors"]
    _write_recon3d_report(config, report)
    return report


def _run_recon3d_frame(
    config: MDICConfig,
    cfg: dict[str, Any],
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    frame_name: str,
    scale: float,
    backend: Any,
) -> dict[str, Any]:
    cam_names = [str(name) for name in cameras["cam_names"]]
    dic_fields = _load_dic2d_fields(cfg["dic2d_dir"], cam_names, frame_name)
    pair_outputs: list[dict[str, Any]] = []
    pair_selection: dict[str, Any] = {}
    if bool(cfg["pair_surface_enabled"]):
        pair_outputs, pair_selection = _run_pair_surface_frame(
            config, cfg, cameras, observations, dic_fields, cam_names, frame_name, scale, backend
        )
    if backend is not None:
        result = backend.reconstruct_tracks(
            np.asarray(cameras["K"], dtype=np.float64),
            np.asarray(cameras["dist"], dtype=np.float64),
            np.asarray(cameras["R"], dtype=np.float64),
            np.asarray(cameras["t"], dtype=np.float64),
            np.asarray(observations["point_indices"], dtype=np.int64),
            np.asarray(observations["cam_indices"], dtype=np.int32),
            np.asarray(observations["uv"], dtype=np.float64),
            dic_fields["u"],
            dic_fields["v"],
            dic_fields["corrcoef"],
            dic_fields["valid"],
            int(dic_fields["reduced_height"]),
            int(dic_fields["reduced_width"]),
            int(dic_fields["subset_spacing"]),
            int(cfg["min_views"]),
            float(cfg["min_corrcoef"]),
            float(cfg["max_reprojection_error_px"]),
            float(scale),
        )
    else:
        result = _reconstruct_tracks_numpy(cameras, observations, dic_fields, cfg, scale)

    stem = Path(frame_name).stem
    npz_path = cfg["output_dir"] / f"recon3d_{stem}.npz"
    np.savez_compressed(npz_path, **result)
    ply_path = None
    if bool(cfg["export_ply"]):
        ply_path = cfg["output_dir"] / f"recon3d_{stem}.ply"
        _write_displacement_ply(ply_path, result["points_ref_world"], result["displacement_world"], result["valid"])
    qc_dir = cfg["output_dir"] / "qc" / stem
    qc_outputs: dict[str, Any] = {}
    qc_summary: dict[str, Any] = {}
    if bool(cfg["qc_enabled"]):
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_summary = _build_recon3d_qc_summary(result, observations, cam_names)
        qc_outputs = _write_recon3d_visualizations(qc_dir, stem, result, qc_summary, cfg)

    valid = np.asarray(result["valid"], dtype=bool)
    disp = np.asarray(result["displacement_world"], dtype=np.float64)
    disp_norm = np.linalg.norm(disp[valid], axis=1) if np.any(valid) else np.asarray([], dtype=np.float64)
    frame_report = {
        "ok": bool(np.count_nonzero(valid) > 0),
        "frame": frame_name,
        "output_npz": str(npz_path),
        "output_ply": str(ply_path) if ply_path else None,
        "num_tracks_total": int(len(valid)),
        "num_tracks_valid": int(np.count_nonzero(valid)),
        "valid_ratio": float(np.count_nonzero(valid) / len(valid)) if len(valid) else 0.0,
        "scale": float(scale),
        "min_views": int(cfg["min_views"]),
        "min_corrcoef": float(cfg["min_corrcoef"]),
        "max_reprojection_error_px": float(cfg["max_reprojection_error_px"]),
        "mean_views": float(np.mean(result["num_views"][valid])) if np.any(valid) else 0.0,
        "mean_corrcoef": float(np.mean(result["mean_corrcoef"][valid])) if np.any(valid) else 0.0,
        "median_displacement_norm": float(np.median(disp_norm)) if disp_norm.size else 0.0,
        "qc": qc_summary,
        "qc_outputs": qc_outputs,
        "pair_selection": pair_selection,
        "pair_surfaces": pair_outputs,
    }
    if not frame_report["ok"]:
        frame_report["error"] = "No valid reconstructed tracks passed the configured filters."
    return frame_report


def _run_pair_surface_frame(
    config: MDICConfig,
    cfg: dict[str, Any],
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    dic_fields: dict[str, np.ndarray],
    cam_names: list[str],
    frame_name: str,
    scale: float,
    backend: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = _select_camera_pairs(cam_names, cameras, observations, cfg)
    selection = _camera_pair_selection_summary(cam_names, cameras, observations, cfg, pairs)
    pair_dir = cfg["output_dir"] / "pairs" / Path(frame_name).stem
    pair_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for pair_index, (cam_a, cam_b) in enumerate(pairs):
        pair_result = _reconstruct_pair_surface(cameras, observations, dic_fields, cfg, cam_a, cam_b, scale, backend)
        pair_name = f"pair_{cam_names[cam_a]}_{cam_names[cam_b]}_{Path(frame_name).stem}"
        npz_path = pair_dir / f"{pair_name}.npz"
        np.savez_compressed(npz_path, **pair_result)
        post_path = None
        post_result: dict[str, np.ndarray] | None = None
        if bool(cfg["post3d_enabled"]):
            post_dir = cfg["output_dir"] / "post" / Path(frame_name).stem
            post_dir.mkdir(parents=True, exist_ok=True)
            post_result = _compute_pair_post3d(pair_result, cfg, backend)
            post_path = post_dir / f"{pair_name}_post.npz"
            np.savez_compressed(post_path, **post_result)
        ply_path = None
        if bool(cfg["export_ply"]):
            ply_path = pair_dir / f"{pair_name}.ply"
            _write_pair_surface_ply(
                ply_path,
                pair_result["faces"],
                pair_result["points_ref_world"],
                pair_result["displacement_world"],
                pair_result["valid_points"],
                pair_result["valid_faces"],
            )
        outputs.append(
            {
                "pair_index": int(pair_index),
                "camera_pair": [cam_names[cam_a], cam_names[cam_b]],
                "output_npz": str(npz_path),
                "output_ply": str(ply_path) if ply_path else None,
                "post_npz": str(post_path) if post_path else None,
                "num_points": int(len(pair_result["point_indices"])),
                "num_valid_points": int(np.count_nonzero(pair_result["valid_points"])),
                "num_faces": int(len(pair_result["faces"])),
                "num_valid_faces": int(np.count_nonzero(pair_result["valid_faces"])),
                "mean_corr_comb": float(np.mean(pair_result["corr_comb"][pair_result["valid_points"]]))
                if np.any(pair_result["valid_points"])
                else 0.0,
                "median_displacement_arbm_norm": float(
                    np.median(post_result["displacement_arbm_norm_world"][pair_result["valid_points"]])
                )
                if post_result is not None and np.any(pair_result["valid_points"])
                else None,
                "median_Eeq": float(np.nanmedian(post_result["Eeq"][post_result["valid_strain_faces"]]))
                if post_result is not None and "Eeq" in post_result and np.any(post_result["valid_strain_faces"])
                else None,
            }
        )
    return outputs, selection


def _reconstruct_pair_surface(
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    dic_fields: dict[str, np.ndarray],
    cfg: dict[str, Any],
    cam_a: int,
    cam_b: int,
    scale: float,
    backend: Any = None,
) -> dict[str, np.ndarray]:
    K = np.asarray(cameras["K"], dtype=np.float64)
    dist = np.asarray(cameras["dist"], dtype=np.float64)
    R = np.asarray(cameras["R"], dtype=np.float64)
    t = np.asarray(cameras["t"], dtype=np.float64)
    point_indices = np.asarray(observations["point_indices"], dtype=np.int64)
    cam_indices = np.asarray(observations["cam_indices"], dtype=np.int32)
    uv_all = np.asarray(observations["uv"], dtype=np.float64)

    if backend is not None and hasattr(backend, "reconstruct_pair_surface_points"):
        point_result = backend.reconstruct_pair_surface_points(
            K,
            dist,
            R,
            t,
            point_indices,
            cam_indices,
            uv_all,
            dic_fields["u"],
            dic_fields["v"],
            dic_fields["corrcoef"],
            dic_fields["valid"],
            int(dic_fields["reduced_height"]),
            int(dic_fields["reduced_width"]),
            int(dic_fields["subset_spacing"]),
            int(cam_a),
            int(cam_b),
            float(cfg["min_corrcoef"]),
            float(cfg["max_reprojection_error_px"]),
            float(scale),
        )
        point_result = {str(key): np.asarray(value) for key, value in point_result.items()}
        if len(point_result["point_indices"]) == 0:
            return _empty_pair_surface(cam_a, cam_b)
        return _finish_pair_surface_from_points(point_result, cfg, cam_a, cam_b)

    by_pair: dict[int, dict[int, int]] = defaultdict(dict)
    for obs_idx, (point_idx, cam_idx) in enumerate(zip(point_indices, cam_indices)):
        if int(cam_idx) in (cam_a, cam_b):
            by_pair[int(point_idx)][int(cam_idx)] = int(obs_idx)

    rows: list[dict[str, Any]] = []
    for point_idx in sorted(by_pair):
        obs_by_cam = by_pair[point_idx]
        if cam_a not in obs_by_cam or cam_b not in obs_by_cam:
            continue
        obs_a = obs_by_cam[cam_a]
        obs_b = obs_by_cam[cam_b]
        sample_a = _sample_dic2d_at(dic_fields, cam_a, uv_all[obs_a], float(cfg["min_corrcoef"]))
        sample_b = _sample_dic2d_at(dic_fields, cam_b, uv_all[obs_b], float(cfg["min_corrcoef"]))
        if sample_a is None or sample_b is None:
            continue
        ua, va, corr_a = sample_a
        ub, vb, corr_b = sample_b
        uv_ref_a = uv_all[obs_a].astype(np.float64)
        uv_ref_b = uv_all[obs_b].astype(np.float64)
        uv_def_a = uv_ref_a + np.asarray([ua, va], dtype=np.float64)
        uv_def_b = uv_ref_b + np.asarray([ub, vb], dtype=np.float64)
        ray_ref_a = _pixel_to_normalized_ray(uv_ref_a, K[cam_a], dist[cam_a])
        ray_ref_b = _pixel_to_normalized_ray(uv_ref_b, K[cam_b], dist[cam_b])
        ray_def_a = _pixel_to_normalized_ray(uv_def_a, K[cam_a], dist[cam_a])
        ray_def_b = _pixel_to_normalized_ray(uv_def_b, K[cam_b], dist[cam_b])
        cams = np.asarray([cam_a, cam_b], dtype=np.int32)
        x_ref = _triangulate_normalized_multiview(np.stack([ray_ref_a, ray_ref_b]), R[cams], t[cams])
        x_def = _triangulate_normalized_multiview(np.stack([ray_def_a, ray_def_b]), R[cams], t[cams])
        if x_ref is None or x_def is None:
            continue
        err_ref = _mean_reprojection_error(x_ref, np.stack([uv_ref_a, uv_ref_b]), K[cams], dist[cams], R[cams], t[cams])
        err_def = _mean_reprojection_error(x_def, np.stack([uv_def_a, uv_def_b]), K[cams], dist[cams], R[cams], t[cams])
        corr_comb = min(float(corr_a), float(corr_b))
        valid_point = (
            corr_comb >= float(cfg["min_corrcoef"])
            and err_ref <= float(cfg["max_reprojection_error_px"])
            and err_def <= float(cfg["max_reprojection_error_px"])
        )
        rows.append(
            {
                "point_idx": int(point_idx),
                "uv_ref_a": uv_ref_a,
                "uv_ref_b": uv_ref_b,
                "uv_def_a": uv_def_a,
                "uv_def_b": uv_def_b,
                "x_ref": x_ref,
                "x_def": x_def,
                "corr_a": float(corr_a),
                "corr_b": float(corr_b),
                "corr_comb": corr_comb,
                "err_ref": float(err_ref),
                "err_def": float(err_def),
                "valid_point": bool(valid_point),
            }
        )

    if not rows:
        return _empty_pair_surface(cam_a, cam_b)

    point_ids = np.asarray([row["point_idx"] for row in rows], dtype=np.int64)
    uv_ref_a = np.stack([row["uv_ref_a"] for row in rows]).astype(np.float64)
    uv_ref_b = np.stack([row["uv_ref_b"] for row in rows]).astype(np.float64)
    uv_def_a = np.stack([row["uv_def_a"] for row in rows]).astype(np.float64)
    uv_def_b = np.stack([row["uv_def_b"] for row in rows]).astype(np.float64)
    points_ref = np.stack([row["x_ref"] for row in rows]).astype(np.float64)
    points_def = np.stack([row["x_def"] for row in rows]).astype(np.float64)
    corr_a = np.asarray([row["corr_a"] for row in rows], dtype=np.float64)
    corr_b = np.asarray([row["corr_b"] for row in rows], dtype=np.float64)
    corr_comb = np.asarray([row["corr_comb"] for row in rows], dtype=np.float64)
    reproj_ref = np.asarray([row["err_ref"] for row in rows], dtype=np.float64)
    reproj_def = np.asarray([row["err_def"] for row in rows], dtype=np.float64)
    valid_points = np.asarray([row["valid_point"] for row in rows], dtype=bool)

    displacement = points_def - points_ref
    point_result = {
        "point_indices": point_ids,
        "uv_ref_a": uv_ref_a,
        "uv_ref_b": uv_ref_b,
        "uv_def_a": uv_def_a,
        "uv_def_b": uv_def_b,
        "points_ref_sfm": points_ref,
        "points_def_sfm": points_def,
        "points_ref_world": points_ref * scale,
        "points_def_world": points_def * scale,
        "displacement_sfm": displacement,
        "displacement_world": displacement * scale,
        "displacement_norm_world": np.linalg.norm(displacement * scale, axis=1),
        "corr_a": corr_a,
        "corr_b": corr_b,
        "corr_comb": corr_comb,
        "reprojection_error_ref": reproj_ref,
        "reprojection_error_def": reproj_def,
        "valid_points": valid_points,
    }
    return _finish_pair_surface_from_points(point_result, cfg, cam_a, cam_b)


def _finish_pair_surface_from_points(
    point_result: dict[str, np.ndarray],
    cfg: dict[str, Any],
    cam_a: int,
    cam_b: int,
) -> dict[str, np.ndarray]:
    uv_ref_a = np.asarray(point_result["uv_ref_a"], dtype=np.float64)
    points_ref_world = np.asarray(point_result["points_ref_world"], dtype=np.float64)
    points_def_world = np.asarray(point_result["points_def_world"], dtype=np.float64)
    corr_comb = np.asarray(point_result["corr_comb"], dtype=np.float64)
    reproj_ref = np.asarray(point_result["reprojection_error_ref"], dtype=np.float64)
    reproj_def = np.asarray(point_result["reprojection_error_def"], dtype=np.float64)
    valid_points = np.asarray(point_result["valid_points"], dtype=bool)
    faces = _triangulate_pair_faces(uv_ref_a, valid_points, float(cfg["pair_max_edge_px"]))
    valid_faces = _filter_pair_faces(
        faces,
        valid_points,
        corr_comb,
        reproj_ref,
        reproj_def,
        uv_ref_a,
        float(cfg["pair_max_edge_px"]),
        float(cfg["pair_min_face_corrcoef"]),
        float(cfg["pair_max_face_reprojection_error_px"]),
    )
    result = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "pair_cam_ids": np.asarray([cam_a, cam_b], dtype=np.int32),
        **point_result,
        "faces": faces.astype(np.int32),
        "valid_faces": valid_faces,
        "face_corr_comb": _face_min(corr_comb, faces),
        "face_centroids_ref": _face_centroids(points_ref_world, faces),
        "face_centroids_def": _face_centroids(points_def_world, faces),
    }
    return {key: np.asarray(value) for key, value in result.items()}


def _empty_pair_surface(cam_a: int, cam_b: int) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "pair_cam_ids": np.asarray([cam_a, cam_b], dtype=np.int32),
        "point_indices": np.zeros(0, dtype=np.int64),
        "uv_ref_a": np.zeros((0, 2), dtype=np.float64),
        "uv_ref_b": np.zeros((0, 2), dtype=np.float64),
        "uv_def_a": np.zeros((0, 2), dtype=np.float64),
        "uv_def_b": np.zeros((0, 2), dtype=np.float64),
        "points_ref_sfm": np.zeros((0, 3), dtype=np.float64),
        "points_def_sfm": np.zeros((0, 3), dtype=np.float64),
        "points_ref_world": np.zeros((0, 3), dtype=np.float64),
        "points_def_world": np.zeros((0, 3), dtype=np.float64),
        "displacement_sfm": np.zeros((0, 3), dtype=np.float64),
        "displacement_world": np.zeros((0, 3), dtype=np.float64),
        "displacement_norm_world": np.zeros(0, dtype=np.float64),
        "corr_a": np.zeros(0, dtype=np.float64),
        "corr_b": np.zeros(0, dtype=np.float64),
        "corr_comb": np.zeros(0, dtype=np.float64),
        "reprojection_error_ref": np.zeros(0, dtype=np.float64),
        "reprojection_error_def": np.zeros(0, dtype=np.float64),
        "valid_points": np.zeros(0, dtype=bool),
        "faces": np.zeros((0, 3), dtype=np.int32),
        "valid_faces": np.zeros(0, dtype=bool),
        "face_corr_comb": np.zeros(0, dtype=np.float64),
        "face_centroids_ref": np.zeros((0, 3), dtype=np.float64),
        "face_centroids_def": np.zeros((0, 3), dtype=np.float64),
    }


def _compute_pair_post3d(pair_result: dict[str, np.ndarray], cfg: dict[str, Any], backend: Any = None) -> dict[str, np.ndarray]:
    points_ref = np.asarray(pair_result["points_ref_world"], dtype=np.float64)
    points_def = np.asarray(pair_result["points_def_world"], dtype=np.float64)
    valid_points = np.asarray(pair_result["valid_points"], dtype=bool)
    faces = np.asarray(pair_result["faces"], dtype=np.int32)
    valid_faces = np.asarray(pair_result["valid_faces"], dtype=bool)
    corr_comb = np.asarray(pair_result["corr_comb"], dtype=np.float64)
    face_corr_comb = np.asarray(pair_result["face_corr_comb"], dtype=np.float64)

    disp = points_def - points_ref
    disp_norm = np.linalg.norm(disp, axis=1)
    if bool(cfg["post3d_remove_rbm"]):
        rot, trans, points_def_arbm = _rigid_transform_points(points_def, points_ref, valid_points)
    else:
        rot = np.eye(3, dtype=np.float64)
        trans = np.zeros(3, dtype=np.float64)
        points_def_arbm = points_def.copy()
    disp_arbm = points_def_arbm - points_ref
    disp_arbm_norm = np.linalg.norm(disp_arbm, axis=1)

    face_centroids_ref = _face_centroids(points_ref, faces)
    face_centroids_def = _face_centroids(points_def, faces)
    face_centroids_arbm = _face_centroids(points_def_arbm, faces)
    face_disp = _face_centroids(disp, faces)
    face_disp_arbm = _face_centroids(disp_arbm, faces)
    face_disp_norm = np.linalg.norm(face_disp, axis=1) if len(face_disp) else np.zeros(0, dtype=np.float64)
    face_disp_arbm_norm = np.linalg.norm(face_disp_arbm, axis=1) if len(face_disp_arbm) else np.zeros(0, dtype=np.float64)
    face_iso = _face_isotropy_index(faces, points_ref)
    face_corr = face_corr_comb if len(face_corr_comb) == len(faces) else _face_min(corr_comb, faces)
    strain: dict[str, np.ndarray] = _empty_surface_deformation(len(faces))
    if bool(cfg["post3d_compute_strain"]):
        if backend is not None and hasattr(backend, "compute_surface_deformation"):
            native_strain = backend.compute_surface_deformation(
                np.asarray(faces, dtype=np.int32),
                np.asarray(points_ref, dtype=np.float64),
                np.asarray(points_def_arbm, dtype=np.float64),
                np.asarray(valid_faces, dtype=bool),
            )
            strain = {str(key): np.asarray(value) for key, value in native_strain.items()}
        else:
            strain = _surface_deformation_numpy(faces, points_ref, points_def_arbm, valid_faces)

    rbm_residual = np.linalg.norm(disp_arbm[valid_points], axis=1) if np.any(valid_points) else np.zeros(0, dtype=np.float64)
    result = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "pair_cam_ids": np.asarray(pair_result["pair_cam_ids"], dtype=np.int32),
        "point_indices": np.asarray(pair_result["point_indices"], dtype=np.int64),
        "faces": faces,
        "valid_points": valid_points,
        "valid_faces": valid_faces,
        "points_ref_world": points_ref,
        "points_def_world": points_def,
        "points_def_arbm_world": points_def_arbm,
        "displacement_world": disp,
        "displacement_norm_world": disp_norm,
        "displacement_arbm_world": disp_arbm,
        "displacement_arbm_norm_world": disp_arbm_norm,
        "rbm_rotation": rot,
        "rbm_translation": trans,
        "rbm_residual_norm_stats": _numeric_stats_array(rbm_residual),
        "face_centroids_ref": face_centroids_ref,
        "face_centroids_def": face_centroids_def,
        "face_centroids_arbm": face_centroids_arbm,
        "face_corr_comb": face_corr,
        "face_isotropy_index": face_iso,
        "face_displacement_world": face_disp,
        "face_displacement_norm_world": face_disp_norm,
        "face_displacement_arbm_world": face_disp_arbm,
        "face_displacement_arbm_norm_world": face_disp_arbm_norm,
    }
    result.update(strain)
    return result


def _rigid_transform_points(points_from: np.ndarray, points_to: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_from = np.asarray(points_from, dtype=np.float64)
    points_to = np.asarray(points_to, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    finite = valid & np.all(np.isfinite(points_from), axis=1) & np.all(np.isfinite(points_to), axis=1)
    if np.count_nonzero(finite) < 3:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), points_from.copy()
    a = points_from[finite]
    b = points_to[finite]
    ac = np.mean(a, axis=0)
    bc = np.mean(b, axis=0)
    da = a - ac
    db = b - bc
    matrix = db.T @ da
    u, _, vh = np.linalg.svd(matrix)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(u @ vh)
    rot = u @ correction @ vh
    trans = bc - rot @ ac
    transformed = (rot @ points_from.T).T + trans
    return rot.astype(np.float64), trans.astype(np.float64), transformed.astype(np.float64)


def _face_isotropy_index(faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32)
    points = np.asarray(points, dtype=np.float64)
    out = np.full(len(faces), np.nan, dtype=np.float64)
    for idx, face in enumerate(faces):
        tri = points[face]
        if not np.all(np.isfinite(tri)):
            continue
        centroid = np.mean(tri, axis=0)
        x = (tri - centroid).T
        k = (x @ x.T) / 3.0
        eigvals = np.sort(np.real(np.linalg.eigvalsh(k)))
        denom = eigvals[1] + eigvals[2]
        out[idx] = 0.0 if abs(float(denom)) <= 1.0e-12 else float(2.0 * eigvals[1] / denom)
    return out


def _empty_surface_deformation(num_faces: int) -> dict[str, np.ndarray]:
    nan = np.nan
    return {
        "Fmat": np.full((num_faces, 3, 3), nan, dtype=np.float64),
        "Cmat": np.full((num_faces, 3, 3), nan, dtype=np.float64),
        "J": np.full(num_faces, nan, dtype=np.float64),
        "Emat": np.full((num_faces, 3, 3), nan, dtype=np.float64),
        "emat": np.full((num_faces, 3, 3), nan, dtype=np.float64),
        "Emgn": np.full(num_faces, nan, dtype=np.float64),
        "emgn": np.full(num_faces, nan, dtype=np.float64),
        "Epc1": np.full(num_faces, nan, dtype=np.float64),
        "Epc2": np.full(num_faces, nan, dtype=np.float64),
        "epc1": np.full(num_faces, nan, dtype=np.float64),
        "epc2": np.full(num_faces, nan, dtype=np.float64),
        "EShearMax": np.full(num_faces, nan, dtype=np.float64),
        "eShearMax": np.full(num_faces, nan, dtype=np.float64),
        "Eeq": np.full(num_faces, nan, dtype=np.float64),
        "eeq": np.full(num_faces, nan, dtype=np.float64),
        "Area": np.full(num_faces, nan, dtype=np.float64),
        "d3": np.full((num_faces, 3), nan, dtype=np.float64),
        "Lambda1": np.full(num_faces, nan, dtype=np.float64),
        "Lambda2": np.full(num_faces, nan, dtype=np.float64),
        "valid_strain_faces": np.zeros(num_faces, dtype=bool),
    }


def _surface_deformation_numpy(
    faces: np.ndarray,
    points_ref: np.ndarray,
    points_def: np.ndarray,
    valid_faces: np.ndarray,
) -> dict[str, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int32)
    points_ref = np.asarray(points_ref, dtype=np.float64)
    points_def = np.asarray(points_def, dtype=np.float64)
    valid_faces = np.asarray(valid_faces, dtype=bool)
    out = _empty_surface_deformation(len(faces))
    eye = np.eye(3, dtype=np.float64)
    for idx, face in enumerate(faces):
        if not valid_faces[idx]:
            continue
        X = points_ref[face]
        x = points_def[face]
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(x)):
            continue
        D1 = X[1] - X[0]
        D2 = X[2] - X[0]
        d1 = x[1] - x[0]
        d2 = x[2] - x[0]
        cross_D = np.cross(D1, D2)
        cross_d = np.cross(d1, d2)
        norm_D = float(np.linalg.norm(cross_D))
        norm_d = float(np.linalg.norm(cross_d))
        if norm_D <= 1.0e-18 or norm_d <= 1.0e-18:
            continue
        D3 = cross_D / norm_D
        d3 = cross_d / norm_d
        Dnorm = float(np.dot(cross_D, D3))
        if abs(Dnorm) <= 1.0e-18:
            continue
        Drec1 = np.cross(D2, D3) / Dnorm
        Drec2 = np.cross(D3, D1) / Dnorm
        F = np.outer(d1, Drec1) + np.outer(d2, Drec2) + np.outer(d3, D3)
        C = F.T @ F
        B = F @ F.T
        try:
            inv_B = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            continue
        E = 0.5 * (C - eye)
        e = 0.5 * (eye - inv_B)
        try:
            lambda_vals = np.sqrt(np.maximum(np.linalg.eigvalsh(C), 0.0))
            eig_E_vals, eig_E_vecs = np.linalg.eigh(E)
            eig_e_vals, eig_e_vecs = np.linalg.eigh(e)
        except np.linalg.LinAlgError:
            continue
        lambda_keep = np.delete(lambda_vals, int(np.argmin(np.abs(lambda_vals - 1.0))))
        lambda_keep.sort()
        E_keep = np.delete(eig_E_vals, int(np.argmax(np.abs(eig_E_vecs.T @ D3))))
        e_keep = np.delete(eig_e_vals, int(np.argmax(np.abs(eig_e_vecs.T @ d3))))
        E_keep.sort()
        e_keep.sort()
        Edev = E - np.trace(E) / 3.0 * eye
        edev = e - np.trace(e) / 3.0 * eye
        out["Fmat"][idx] = F
        out["Cmat"][idx] = C
        out["J"][idx] = np.linalg.det(F)
        out["Emat"][idx] = E
        out["emat"][idx] = e
        out["Emgn"][idx] = np.linalg.norm(E, ord="fro")
        out["emgn"][idx] = np.linalg.norm(e, ord="fro")
        out["Epc1"][idx] = E_keep[0]
        out["Epc2"][idx] = E_keep[1]
        out["epc1"][idx] = e_keep[0]
        out["epc2"][idx] = e_keep[1]
        out["EShearMax"][idx] = 0.5 * (E_keep[1] - E_keep[0])
        out["eShearMax"][idx] = 0.5 * (e_keep[1] - e_keep[0])
        out["Eeq"][idx] = np.sqrt((2.0 / 3.0) * np.sum(Edev * Edev))
        out["eeq"][idx] = np.sqrt((2.0 / 3.0) * np.sum(edev * edev))
        out["Area"][idx] = 0.5 * Dnorm
        out["d3"][idx] = d3
        out["Lambda1"][idx] = lambda_keep[0]
        out["Lambda2"][idx] = lambda_keep[1]
        out["valid_strain_faces"][idx] = True
    return out


def _numeric_stats_array(values: np.ndarray) -> np.ndarray:
    stats = _array_stats(values)
    keys = ("count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max")
    return np.asarray([np.nan if stats[key] is None else float(stats[key]) for key in keys], dtype=np.float64)


def _reconstruct_tracks_numpy(
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    dic_fields: dict[str, np.ndarray],
    cfg: dict[str, Any],
    scale: float,
) -> dict[str, np.ndarray]:
    K = np.asarray(cameras["K"], dtype=np.float64)
    dist = np.asarray(cameras["dist"], dtype=np.float64)
    R = np.asarray(cameras["R"], dtype=np.float64)
    t = np.asarray(cameras["t"], dtype=np.float64)
    point_indices = np.asarray(observations["point_indices"], dtype=np.int64)
    cam_indices = np.asarray(observations["cam_indices"], dtype=np.int32)
    uv = np.asarray(observations["uv"], dtype=np.float64)

    by_track: dict[int, list[int]] = defaultdict(list)
    for obs_idx, point_idx in enumerate(point_indices):
        by_track[int(point_idx)].append(obs_idx)

    track_ids = np.asarray(sorted(by_track.keys()), dtype=np.int64)
    n_tracks = len(track_ids)
    points_ref = np.zeros((n_tracks, 3), dtype=np.float64)
    points_def = np.zeros((n_tracks, 3), dtype=np.float64)
    reproj_ref = np.full(n_tracks, np.inf, dtype=np.float64)
    reproj_def = np.full(n_tracks, np.inf, dtype=np.float64)
    num_views = np.zeros(n_tracks, dtype=np.int32)
    mean_corrcoef = np.zeros(n_tracks, dtype=np.float64)
    valid = np.zeros(n_tracks, dtype=bool)

    for out_idx, track_id in enumerate(track_ids):
        obs_ids = by_track[int(track_id)]
        ref_rays: list[np.ndarray] = []
        def_rays: list[np.ndarray] = []
        cams: list[int] = []
        corrs: list[float] = []
        uv_ref_used: list[np.ndarray] = []
        uv_def_used: list[np.ndarray] = []
        for obs_id in obs_ids:
            cam_id = int(cam_indices[obs_id])
            sample = _sample_dic2d_at(dic_fields, cam_id, uv[obs_id], float(cfg["min_corrcoef"]))
            if sample is None:
                continue
            u, v, corr = sample
            uv_ref = uv[obs_id].astype(np.float64)
            uv_def = uv_ref + np.asarray([u, v], dtype=np.float64)
            ref_rays.append(_pixel_to_normalized_ray(uv_ref, K[cam_id], dist[cam_id]))
            def_rays.append(_pixel_to_normalized_ray(uv_def, K[cam_id], dist[cam_id]))
            cams.append(cam_id)
            corrs.append(corr)
            uv_ref_used.append(uv_ref)
            uv_def_used.append(uv_def)

        if len(cams) < int(cfg["min_views"]):
            continue

        X_ref = _triangulate_normalized_multiview(np.asarray(ref_rays), R[np.asarray(cams)], t[np.asarray(cams)])
        X_def = _triangulate_normalized_multiview(np.asarray(def_rays), R[np.asarray(cams)], t[np.asarray(cams)])
        if X_ref is None or X_def is None:
            continue

        err_ref = _mean_reprojection_error(X_ref, np.asarray(uv_ref_used), K[np.asarray(cams)], dist[np.asarray(cams)], R[np.asarray(cams)], t[np.asarray(cams)])
        err_def = _mean_reprojection_error(X_def, np.asarray(uv_def_used), K[np.asarray(cams)], dist[np.asarray(cams)], R[np.asarray(cams)], t[np.asarray(cams)])
        points_ref[out_idx] = X_ref
        points_def[out_idx] = X_def
        reproj_ref[out_idx] = err_ref
        reproj_def[out_idx] = err_def
        num_views[out_idx] = len(cams)
        mean_corrcoef[out_idx] = float(np.mean(corrs))
        valid[out_idx] = (
            np.isfinite(err_ref)
            and np.isfinite(err_def)
            and err_ref <= float(cfg["max_reprojection_error_px"])
            and err_def <= float(cfg["max_reprojection_error_px"])
        )

    displacement = points_def - points_ref
    return {
        "point_indices": track_ids,
        "points_ref_sfm": points_ref,
        "points_def_sfm": points_def,
        "displacement_sfm": displacement,
        "points_ref_world": points_ref * scale,
        "points_def_world": points_def * scale,
        "displacement_world": displacement * scale,
        "num_views": num_views,
        "mean_corrcoef": mean_corrcoef,
        "reprojection_error_ref": reproj_ref,
        "reprojection_error_def": reproj_def,
        "valid": valid,
    }


def _sample_dic2d_at(dic_fields: dict[str, np.ndarray], cam_id: int, uv: np.ndarray, min_corrcoef: float) -> tuple[float, float, float] | None:
    step = int(dic_fields["subset_spacing"]) + 1
    gx = float(uv[0]) / step
    gy = float(uv[1]) / step
    u = dic_fields["u"][cam_id]
    v = dic_fields["v"][cam_id]
    corr = dic_fields["corrcoef"][cam_id]
    valid = dic_fields["valid"][cam_id]
    sample = _bilinear_sample_fields(gx, gy, u, v, corr, valid, min_corrcoef)
    if sample is not None:
        return sample
    return _nearest_sample_fields(gx, gy, u, v, corr, valid, min_corrcoef)


def _select_camera_pairs(
    cam_names: list[str],
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> list[tuple[int, int]]:
    mode = str(cfg["pair_mode"]).lower()
    if mode == "manual":
        name_to_id = {name: idx for idx, name in enumerate(cam_names)}
        pairs: list[tuple[int, int]] = []
        for item in cfg["manual_pairs"]:
            if len(item) != 2:
                raise ValueError(f"Manual camera pair must contain two entries, got {item!r}.")
            a = _camera_pair_entry_to_id(item[0], name_to_id)
            b = _camera_pair_entry_to_id(item[1], name_to_id)
            if a == b:
                raise ValueError(f"Manual camera pair cannot repeat the same camera: {item!r}.")
            pairs.append((a, b))
        return _deduplicate_pairs(pairs)
    order = sorted(range(len(cam_names)), key=lambda idx: _camera_order_value(cam_names[idx], idx))
    if mode == "adjacent":
        pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
        if bool(cfg["pair_wrap"]) and len(order) > 2:
            pairs.append((order[-1], order[0]))
        return _deduplicate_pairs(pairs)
    if mode == "all":
        return [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    if mode == "auto_spatial":
        return _auto_spatial_camera_pairs(cam_names, cameras, observations, cfg)
    raise ValueError(f"Unknown recon3d.pairs.mode={mode!r}; expected auto_spatial, adjacent, manual, or all.")


def _auto_spatial_camera_pairs(
    cam_names: list[str],
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> list[tuple[int, int]]:
    centers = np.asarray(cameras.get("camera_centers_world"), dtype=np.float64)
    if centers.shape != (len(cam_names), 3) or len(cam_names) < 2:
        order = sorted(range(len(cam_names)), key=lambda idx: _camera_order_value(cam_names[idx], idx))
        return _deduplicate_pairs([(order[i], order[i + 1]) for i in range(len(order) - 1)])

    order, circularity = _spatial_camera_order(centers)
    adjacent_distances = [
        float(np.linalg.norm(centers[order[i + 1]] - centers[order[i]]))
        for i in range(len(order) - 1)
    ]
    wrap_distance = float(np.linalg.norm(centers[order[0]] - centers[order[-1]])) if len(order) > 2 else np.inf
    median_adjacent = float(np.median(adjacent_distances)) if adjacent_distances else wrap_distance
    wrap_ratio = wrap_distance / max(median_adjacent, 1.0e-12)
    shared_counts = _shared_track_counts(observations, len(cam_names))
    adjacent_shared = [shared_counts[order[i], order[i + 1]] for i in range(len(order) - 1)]
    median_shared = float(np.median(adjacent_shared)) if adjacent_shared else 0.0
    wrap_shared = int(shared_counts[order[-1], order[0]]) if len(order) > 2 else 0

    circular = (
        circularity >= float(cfg["auto_circularity_threshold"])
        and wrap_ratio <= float(cfg["auto_wrap_distance_ratio"])
        and wrap_shared >= max(int(cfg["auto_min_shared_tracks"]), int(round(median_shared * float(cfg["auto_wrap_min_shared_ratio"]))))
    )
    pairs: list[tuple[int, int]] = []
    max_distance = median_adjacent * float(cfg["auto_max_neighbor_distance_ratio"])
    min_shared = int(cfg["auto_min_shared_tracks"])
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        distance = float(np.linalg.norm(centers[a] - centers[b]))
        if distance <= max_distance and shared_counts[a, b] >= min_shared:
            pairs.append((a, b))
    if circular:
        pairs.append((order[-1], order[0]))
    return _deduplicate_pairs(pairs)


def _spatial_camera_order(centers: np.ndarray) -> tuple[list[int], float]:
    centroid = np.mean(centers, axis=0)
    centered = centers - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    axis0 = vh[0]
    axis1 = vh[1] if vh.shape[0] > 1 else np.asarray([0.0, 1.0, 0.0])
    projected = np.column_stack([centered @ axis0, centered @ axis1])
    radial = np.linalg.norm(projected, axis=1)
    circularity = float(np.min(radial) / max(np.max(radial), 1.0e-12)) if radial.size else 0.0
    if circularity >= 0.45 and len(centers) >= 4:
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        return [int(idx) for idx in np.argsort(angles)], circularity
    scores = projected[:, 0]
    return [int(idx) for idx in np.argsort(scores)], circularity


def _shared_track_counts(observations: dict[str, np.ndarray], n_cameras: int) -> np.ndarray:
    point_indices = np.asarray(observations["point_indices"], dtype=np.int64)
    cam_indices = np.asarray(observations["cam_indices"], dtype=np.int32)
    by_point: dict[int, set[int]] = defaultdict(set)
    for point_idx, cam_idx in zip(point_indices, cam_indices):
        if 0 <= int(cam_idx) < n_cameras:
            by_point[int(point_idx)].add(int(cam_idx))
    counts = np.zeros((n_cameras, n_cameras), dtype=np.int32)
    for cams in by_point.values():
        cam_list = sorted(cams)
        for i, cam_a in enumerate(cam_list):
            for cam_b in cam_list[i + 1 :]:
                counts[cam_a, cam_b] += 1
                counts[cam_b, cam_a] += 1
    return counts


def _camera_pair_selection_summary(
    cam_names: list[str],
    cameras: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    cfg: dict[str, Any],
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    mode = str(cfg["pair_mode"]).lower()
    summary: dict[str, Any] = {
        "mode": mode,
        "pairs": [[cam_names[a], cam_names[b]] for a, b in pairs],
    }
    centers = np.asarray(cameras.get("camera_centers_world"), dtype=np.float64)
    if centers.shape != (len(cam_names), 3):
        return summary
    order, circularity = _spatial_camera_order(centers)
    shared_counts = _shared_track_counts(observations, len(cam_names))
    adjacent_distances = [
        float(np.linalg.norm(centers[order[i + 1]] - centers[order[i]]))
        for i in range(len(order) - 1)
    ]
    adjacent_shared = [int(shared_counts[order[i], order[i + 1]]) for i in range(len(order) - 1)]
    wrap_distance = float(np.linalg.norm(centers[order[0]] - centers[order[-1]])) if len(order) > 2 else None
    wrap_shared = int(shared_counts[order[-1], order[0]]) if len(order) > 2 else None
    median_adjacent = float(np.median(adjacent_distances)) if adjacent_distances else None
    median_shared = float(np.median(adjacent_shared)) if adjacent_shared else None
    wrap_ratio = float(wrap_distance / max(median_adjacent, 1.0e-12)) if wrap_distance is not None and median_adjacent is not None else None
    circular = (
        circularity >= float(cfg["auto_circularity_threshold"])
        and wrap_ratio is not None
        and wrap_ratio <= float(cfg["auto_wrap_distance_ratio"])
        and wrap_shared is not None
        and median_shared is not None
        and wrap_shared >= max(int(cfg["auto_min_shared_tracks"]), int(round(median_shared * float(cfg["auto_wrap_min_shared_ratio"]))))
    )
    summary.update(
        {
            "spatial_order": [cam_names[idx] for idx in order],
            "circularity": float(circularity),
            "is_circular": bool(circular),
            "median_adjacent_distance": median_adjacent,
            "wrap_distance": wrap_distance,
            "wrap_distance_ratio": wrap_ratio,
            "median_adjacent_shared_tracks": median_shared,
            "wrap_shared_tracks": wrap_shared,
            "thresholds": {
                "auto_circularity_threshold": float(cfg["auto_circularity_threshold"]),
                "auto_wrap_distance_ratio": float(cfg["auto_wrap_distance_ratio"]),
                "auto_wrap_min_shared_ratio": float(cfg["auto_wrap_min_shared_ratio"]),
                "auto_max_neighbor_distance_ratio": float(cfg["auto_max_neighbor_distance_ratio"]),
                "auto_min_shared_tracks": int(cfg["auto_min_shared_tracks"]),
            },
        }
    )
    return summary


def _camera_pair_entry_to_id(value: Any, name_to_id: dict[str, int]) -> int:
    if isinstance(value, int):
        return int(value)
    text = str(value)
    if text in name_to_id:
        return name_to_id[text]
    if text.isdigit():
        return int(text)
    raise ValueError(f"Unknown camera pair entry: {value!r}.")


def _deduplicate_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for a, b in pairs:
        key = tuple(sorted((int(a), int(b))))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(a), int(b)))
    return out


def _camera_order_value(name: str, fallback: int) -> int:
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else fallback


def _triangulate_pair_faces(uv_ref: np.ndarray, valid_points: np.ndarray, max_edge_px: float) -> np.ndarray:
    valid_indices = np.flatnonzero(valid_points)
    if len(valid_indices) < 3:
        return np.zeros((0, 3), dtype=np.int32)
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise RuntimeError("scipy is required for pair-surface Delaunay triangulation.") from exc
    tri = Delaunay(uv_ref[valid_indices])
    faces = valid_indices[np.asarray(tri.simplices, dtype=np.int64)]
    if max_edge_px <= 0.0:
        return faces.astype(np.int32)
    keep = _face_max_edge_px(uv_ref, faces) <= max_edge_px
    return faces[keep].astype(np.int32)


def _filter_pair_faces(
    faces: np.ndarray,
    valid_points: np.ndarray,
    corr_comb: np.ndarray,
    reproj_ref: np.ndarray,
    reproj_def: np.ndarray,
    uv_ref: np.ndarray,
    max_edge_px: float,
    min_face_corrcoef: float,
    max_face_reprojection_error_px: float,
) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    valid = np.all(valid_points[faces], axis=1)
    if max_edge_px > 0.0:
        valid &= _face_max_edge_px(uv_ref, faces) <= max_edge_px
    valid &= np.min(corr_comb[faces], axis=1) >= min_face_corrcoef
    valid &= np.max(reproj_ref[faces], axis=1) <= max_face_reprojection_error_px
    valid &= np.max(reproj_def[faces], axis=1) <= max_face_reprojection_error_px
    return valid.astype(bool)


def _face_max_edge_px(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = uv[faces[:, 0]]
    p1 = uv[faces[:, 1]]
    p2 = uv[faces[:, 2]]
    e01 = np.linalg.norm(p0 - p1, axis=1)
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e20 = np.linalg.norm(p2 - p0, axis=1)
    return np.maximum(np.maximum(e01, e12), e20)


def _face_min(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.min(values[faces], axis=1).astype(np.float64)


def _face_centroids(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.mean(points[faces], axis=1).astype(np.float64)


def _bilinear_sample_fields(
    gx: float,
    gy: float,
    u: np.ndarray,
    v: np.ndarray,
    corr: np.ndarray,
    valid: np.ndarray,
    min_corrcoef: float,
) -> tuple[float, float, float] | None:
    h, w = valid.shape
    x0 = int(np.floor(gx))
    y0 = int(np.floor(gy))
    x1 = x0 + 1
    y1 = y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None
    ys = np.asarray([y0, y0, y1, y1])
    xs = np.asarray([x0, x1, x0, x1])
    if not np.all(valid[ys, xs]):
        return None
    corr_values = corr[ys, xs]
    if float(np.min(corr_values)) < min_corrcoef:
        return None
    wx = gx - x0
    wy = gy - y0
    weights = np.asarray([(1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy], dtype=np.float64)
    return (
        float(np.sum(u[ys, xs] * weights)),
        float(np.sum(v[ys, xs] * weights)),
        float(np.sum(corr_values * weights)),
    )


def _nearest_sample_fields(
    gx: float,
    gy: float,
    u: np.ndarray,
    v: np.ndarray,
    corr: np.ndarray,
    valid: np.ndarray,
    min_corrcoef: float,
) -> tuple[float, float, float] | None:
    h, w = valid.shape
    x = int(round(gx))
    y = int(round(gy))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    if not bool(valid[y, x]) or float(corr[y, x]) < min_corrcoef:
        return None
    return float(u[y, x]), float(v[y, x]), float(corr[y, x])


def _pixel_to_normalized_ray(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    x = (float(uv[0]) - float(K[0, 2])) / float(K[0, 0])
    y = (float(uv[1]) - float(K[1, 2])) / float(K[1, 1])
    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    if abs(k1) > 1e-12 or abs(k2) > 1e-12:
        xu, yu = x, y
        for _ in range(8):
            r2 = xu * xu + yu * yu
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            if abs(radial) <= 1e-12:
                break
            xu = x / radial
            yu = y / radial
        x, y = xu, yu
    return np.asarray([x, y], dtype=np.float64)


def _triangulate_normalized_multiview(rays: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    if len(rays) < 2:
        return None
    rows = []
    for ray, rot, trans in zip(rays, R, t):
        p = np.concatenate([rot, trans.reshape(3, 1)], axis=1)
        rows.append(ray[0] * p[2] - p[0])
        rows.append(ray[1] * p[2] - p[1])
    a = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    homog = vh[-1]
    if abs(float(homog[3])) <= 1e-12:
        return None
    xyz = homog[:3] / homog[3]
    return xyz if np.all(np.isfinite(xyz)) else None


def _mean_reprojection_error(
    point: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    projected = _project_points_multicam(point, K, dist, R, t)
    return float(np.mean(np.linalg.norm(projected - uv, axis=1)))


def _project_points_multicam(point: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    projected = np.zeros((len(R), 2), dtype=np.float64)
    for idx, (k, d, rot, trans) in enumerate(zip(K, dist, R, t)):
        cam = rot @ point + trans
        if abs(float(cam[2])) <= 1e-12:
            projected[idx] = np.nan
            continue
        x = cam[0] / cam[2]
        y = cam[1] / cam[2]
        k1 = float(d[0]) if len(d) > 0 else 0.0
        k2 = float(d[1]) if len(d) > 1 else 0.0
        if abs(k1) > 1e-12 or abs(k2) > 1e-12:
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            x *= radial
            y *= radial
        projected[idx, 0] = k[0, 0] * x + k[0, 2]
        projected[idx, 1] = k[1, 1] * y + k[1, 2]
    return projected


def _load_dic2d_fields(dic2d_dir: Path, cam_names: list[str], frame_name: str) -> dict[str, np.ndarray]:
    stem = Path(frame_name).stem
    stacks: dict[str, list[np.ndarray]] = {key: [] for key in ("u", "v", "corrcoef", "valid")}
    reduced_width = reduced_height = subset_spacing = None
    for cam_name in cam_names:
        path = dic2d_dir / f"dic2d_{cam_name}_{stem}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            schema = int(data["output_schema_version"]) if "output_schema_version" in data.files else 1
            if schema < 2:
                raise ValueError(f"{path} uses DIC2D schema {schema}; recon3d requires schema 2.")
            for key in stacks:
                stacks[key].append(np.asarray(data[key]).copy())
            reduced_width = int(data["reduced_width"])
            reduced_height = int(data["reduced_height"])
            subset_spacing = int(data["subset_spacing"])
    return {
        "u": np.stack(stacks["u"], axis=0).astype(np.float64),
        "v": np.stack(stacks["v"], axis=0).astype(np.float64),
        "corrcoef": np.stack(stacks["corrcoef"], axis=0).astype(np.float64),
        "valid": np.stack(stacks["valid"], axis=0).astype(bool),
        "reduced_width": int(reduced_width),
        "reduced_height": int(reduced_height),
        "subset_spacing": int(subset_spacing),
    }


def _write_displacement_ply(path: Path, points: np.ndarray, displacement: np.ndarray, valid: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    valid_points = points[valid]
    valid_disp = displacement[valid]
    norms = np.linalg.norm(valid_disp, axis=1)
    max_norm = float(np.max(norms)) if norms.size else 1.0
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(valid_points)}\n")
        handle.write("property double x\nproperty double y\nproperty double z\n")
        handle.write("property double ux\nproperty double uy\nproperty double uz\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, disp, norm in zip(valid_points, valid_disp, norms):
            value = 0 if max_norm <= 0.0 else int(np.clip(round(255.0 * norm / max_norm), 0, 255))
            handle.write(
                f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} "
                f"{disp[0]:.10g} {disp[1]:.10g} {disp[2]:.10g} "
                f"{value} {64} {255 - value}\n"
            )


def _write_pair_surface_ply(
    path: Path,
    faces: np.ndarray,
    points: np.ndarray,
    displacement: np.ndarray,
    valid_points: np.ndarray,
    valid_faces: np.ndarray,
) -> None:
    faces = np.asarray(faces, dtype=np.int32)
    points = np.asarray(points, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    valid_points = np.asarray(valid_points, dtype=bool)
    valid_faces = np.asarray(valid_faces, dtype=bool)
    used_faces = faces[valid_faces]
    used_vertices = np.unique(used_faces.reshape(-1)) if len(used_faces) else np.zeros(0, dtype=np.int32)
    used_vertices = used_vertices[valid_points[used_vertices]] if len(used_vertices) else used_vertices
    old_to_new = {int(old): idx for idx, old in enumerate(used_vertices)}
    remapped_faces = []
    for face in used_faces:
        if all(int(idx) in old_to_new for idx in face):
            remapped_faces.append([old_to_new[int(face[0])], old_to_new[int(face[1])], old_to_new[int(face[2])]])
    remapped = np.asarray(remapped_faces, dtype=np.int32)
    verts = points[used_vertices]
    disp_norm = np.linalg.norm(displacement[used_vertices], axis=1) if len(used_vertices) else np.zeros(0, dtype=np.float64)
    max_norm = float(np.max(disp_norm)) if disp_norm.size else 1.0
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(verts)}\n")
        handle.write("property double x\nproperty double y\nproperty double z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(remapped)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for point, norm in zip(verts, disp_norm):
            red, green, blue = _blue_red_colormap(norm, max_norm)
            handle.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} {red} {green} {blue}\n")
        for face in remapped:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _build_recon3d_qc_summary(
    result: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    cam_names: list[str],
) -> dict[str, Any]:
    valid = np.asarray(result["valid"], dtype=bool)
    disp = np.asarray(result["displacement_world"], dtype=np.float64)
    disp_norm = np.linalg.norm(disp, axis=1)
    summary = {
        "displacement_norm": _array_stats(disp_norm[valid]),
        "reprojection_error_ref": _array_stats(np.asarray(result["reprojection_error_ref"], dtype=np.float64)[valid]),
        "reprojection_error_def": _array_stats(np.asarray(result["reprojection_error_def"], dtype=np.float64)[valid]),
        "mean_corrcoef": _array_stats(np.asarray(result["mean_corrcoef"], dtype=np.float64)[valid]),
        "num_views": _array_stats(np.asarray(result["num_views"], dtype=np.float64)[valid]),
        "num_views_histogram": _integer_histogram(np.asarray(result["num_views"], dtype=np.int32)[valid]),
        "camera_contributions": _camera_contribution_summary(result, observations, cam_names),
    }
    return summary


def _array_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _integer_histogram(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique, counts)}


def _camera_contribution_summary(
    result: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    cam_names: list[str],
) -> list[dict[str, Any]]:
    track_ids = np.asarray(result["point_indices"], dtype=np.int64)
    valid = np.asarray(result["valid"], dtype=bool)
    valid_tracks = set(int(track_id) for track_id in track_ids[valid])
    all_tracks = set(int(track_id) for track_id in track_ids)
    point_indices = np.asarray(observations["point_indices"], dtype=np.int64)
    cam_indices = np.asarray(observations["cam_indices"], dtype=np.int32)
    rows: list[dict[str, Any]] = []
    for cam_id, cam_name in enumerate(cam_names):
        cam_mask = cam_indices == cam_id
        cam_track_ids = [int(value) for value in point_indices[cam_mask]]
        total_track_count = sum(1 for track_id in cam_track_ids if track_id in all_tracks)
        valid_track_count = sum(1 for track_id in cam_track_ids if track_id in valid_tracks)
        rows.append(
            {
                "cam_id": int(cam_id),
                "cam_name": cam_name,
                "observed_tracks": int(total_track_count),
                "valid_reconstructed_tracks": int(valid_track_count),
                "valid_ratio": float(valid_track_count / total_track_count) if total_track_count else 0.0,
            }
        )
    return rows


def _write_recon3d_visualizations(
    qc_dir: Path,
    stem: str,
    result: dict[str, np.ndarray],
    qc_summary: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    valid = np.asarray(result["valid"], dtype=bool)
    disp = np.asarray(result["displacement_world"], dtype=np.float64)
    disp_norm = np.linalg.norm(disp, axis=1)
    ref_error = np.asarray(result["reprojection_error_ref"], dtype=np.float64)
    def_error = np.asarray(result["reprojection_error_def"], dtype=np.float64)
    mean_corr = np.asarray(result["mean_corrcoef"], dtype=np.float64)
    num_views = np.asarray(result["num_views"], dtype=np.int32)

    if bool(cfg["export_png"]):
        outputs["displacement_norm_histogram"] = str(
            _save_histogram(qc_dir / f"{stem}_displacement_norm_hist.png", disp_norm[valid], "3D displacement norm")
        )
        outputs["reprojection_error_histogram"] = str(
            _save_two_histogram(
                qc_dir / f"{stem}_reprojection_error_hist.png",
                ref_error[valid],
                def_error[valid],
                "Reprojection error",
                "reference",
                "deformed",
            )
        )
        outputs["corrcoef_histogram"] = str(
            _save_histogram(qc_dir / f"{stem}_corrcoef_hist.png", mean_corr[valid], "Mean DIC correlation")
        )
        outputs["num_views_histogram"] = str(
            _save_integer_bar(qc_dir / f"{stem}_num_views_hist.png", num_views[valid], "Views per track")
        )
        outputs["camera_contributions"] = str(
            _save_camera_contributions(
                qc_dir / f"{stem}_camera_contributions.png",
                qc_summary.get("camera_contributions", []),
            )
        )

    if bool(cfg["export_qc_ply"]):
        points_ply = qc_dir / f"{stem}_points_ref_colored.ply"
        vectors_ply = qc_dir / f"{stem}_displacement_vectors.ply"
        _write_points_ref_ply(points_ply, result["points_ref_world"], disp_norm, valid)
        _write_vector_ply(vectors_ply, result["points_ref_world"], result["displacement_world"], valid, float(cfg["vector_scale"]))
        outputs["points_ref_colored_ply"] = str(points_ply)
        outputs["displacement_vectors_ply"] = str(vectors_ply)

    return outputs


def _save_histogram(path: Path, values: np.ndarray, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.hist(values, bins=_hist_bins(values), color="#2f6f9f", edgecolor="#ffffff", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_two_histogram(path: Path, a: np.ndarray, b: np.ndarray, title: str, label_a: str, label_b: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    merged = np.concatenate([a, b]) if a.size or b.size else np.asarray([], dtype=np.float64)
    bins = _hist_bins(merged)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.hist(a, bins=bins, alpha=0.68, label=label_a, color="#2f6f9f", edgecolor="#ffffff", linewidth=0.4)
    ax.hist(b, bins=bins, alpha=0.58, label=label_b, color="#c44e52", edgecolor="#ffffff", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_integer_bar(path: Path, values: np.ndarray, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = _integer_histogram(values)
    labels = list(hist.keys())
    counts = list(hist.values())
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.bar(labels, counts, color="#447c69")
    ax.set_title(title)
    ax.set_xlabel("Views")
    ax.set_ylabel("Track count")
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_camera_contributions(path: Path, rows: list[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["cam_name"]) for row in rows]
    valid_counts = [int(row["valid_reconstructed_tracks"]) for row in rows]
    total_counts = [int(row["observed_tracks"]) for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=160)
    ax.bar(x, total_counts, color="#d8dee9", label="observed")
    ax.bar(x, valid_counts, color="#447c69", label="valid")
    ax.set_title("Camera contributions")
    ax.set_xlabel("Camera")
    ax.set_ylabel("Track count")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _hist_bins(values: np.ndarray) -> int | np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1
    return min(80, max(12, int(np.sqrt(values.size))))


def _write_points_ref_ply(path: Path, points: np.ndarray, displacement_norm: np.ndarray, valid: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    displacement_norm = np.asarray(displacement_norm, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    valid_points = points[valid]
    valid_norm = displacement_norm[valid]
    max_norm = float(np.max(valid_norm)) if valid_norm.size else 1.0
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(valid_points)}\n")
        handle.write("property double x\nproperty double y\nproperty double z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, norm in zip(valid_points, valid_norm):
            red, green, blue = _blue_red_colormap(norm, max_norm)
            handle.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} {red} {green} {blue}\n")


def _write_vector_ply(path: Path, points: np.ndarray, displacement: np.ndarray, valid: np.ndarray, vector_scale: float) -> None:
    points = np.asarray(points, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    start = points[valid]
    vectors = displacement[valid] * float(vector_scale)
    end = start + vectors
    vertices = np.concatenate([start, end], axis=0) if len(start) else np.zeros((0, 3), dtype=np.float64)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property double x\nproperty double y\nproperty double z\n")
        handle.write(f"element edge {len(start)}\n")
        handle.write("property int vertex1\nproperty int vertex2\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point in vertices:
            handle.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")
        for idx in range(len(start)):
            handle.write(f"{idx} {idx + len(start)} 218 84 45\n")


def _blue_red_colormap(value: float, max_value: float) -> tuple[int, int, int]:
    ratio = 0.0 if max_value <= 0.0 else float(np.clip(value / max_value, 0.0, 1.0))
    red = int(round(40 + 205 * ratio))
    green = int(round(90 + 70 * (1.0 - abs(2.0 * ratio - 1.0))))
    blue = int(round(215 - 175 * ratio))
    return red, green, blue


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_sfm_to_world_scale(config: MDICConfig, cfg: dict[str, Any]) -> float:
    if not bool(cfg["use_scale_correction"]):
        return 1.0
    scale_path = config.result_root / str(cfg["scale_dir"]) / "sfm2world_scale.json"
    if not scale_path.exists():
        return 1.0
    with scale_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("scale", {}).get("sfm_to_world_scale", 1.0))


def _native_backend() -> Any:
    try:
        import native_recon3d  # type: ignore[import-not-found]
    except Exception:
        return None
    return native_recon3d


def _recon_config(config: MDICConfig) -> dict[str, Any]:
    raw = config.raw.get("recon3d", config.raw.get("reconstruction", {}))
    if not isinstance(raw, dict):
        raw = {}
    colmap_cfg = config.raw.get("colmap", {})
    workspace = str(colmap_cfg.get("workspace", "colmap")) if isinstance(colmap_cfg, dict) else "colmap"
    export = raw.get("export", {})
    if not isinstance(export, dict):
        export = {}
    qc = raw.get("qc", {})
    if not isinstance(qc, dict):
        qc = {}
    pairs = raw.get("pairs", {})
    if not isinstance(pairs, dict):
        pairs = {}
    pair_surface = raw.get("pair_surface", {})
    if not isinstance(pair_surface, dict):
        pair_surface = {}
    post3d = raw.get("post3d", {})
    if not isinstance(post3d, dict):
        post3d = {}
    return {
        "sfm_dir": config.result_root / "sfm" / workspace,
        "dic2d_dir": config.result_root / str(raw.get("input_dic2d_dir", "dic2d")),
        "output_dir": config.result_root / str(raw.get("output_dir", "recon3d")),
        "scale_dir": str(raw.get("scale_dir", "scale")),
        "prefer_native": str(raw.get("backend", "native")).lower() == "native",
        "min_views": int(raw.get("min_views", 2)),
        "min_corrcoef": float(raw.get("min_corrcoef", 0.6)),
        "max_reprojection_error_px": float(raw.get("max_reprojection_error_px", 2.0)),
        "use_scale_correction": bool(raw.get("use_scale_correction", True)),
        "export_ply": bool(export.get("ply", True)),
        "export_png": bool(export.get("png", qc.get("plots", True))),
        "export_qc_ply": bool(export.get("qc_ply", True)),
        "qc_enabled": bool(qc.get("enabled", True)),
        "vector_scale": float(qc.get("vector_scale", 1.0)),
        "pair_mode": str(pairs.get("mode", "auto_spatial")),
        "manual_pairs": pairs.get("manual", []),
        "pair_wrap": bool(pairs.get("wrap", True)),
        "auto_circularity_threshold": float(pairs.get("auto_circularity_threshold", 0.45)),
        "auto_wrap_distance_ratio": float(pairs.get("auto_wrap_distance_ratio", 1.8)),
        "auto_wrap_min_shared_ratio": float(pairs.get("auto_wrap_min_shared_ratio", 0.35)),
        "auto_max_neighbor_distance_ratio": float(pairs.get("auto_max_neighbor_distance_ratio", 2.0)),
        "auto_min_shared_tracks": int(pairs.get("auto_min_shared_tracks", 20)),
        "pair_surface_enabled": bool(pair_surface.get("enabled", True)),
        "pair_max_edge_px": float(pair_surface.get("max_edge_px", 80.0)),
        "pair_min_face_corrcoef": float(pair_surface.get("min_face_corrcoef", raw.get("min_corrcoef", 0.6))),
        "pair_max_face_reprojection_error_px": float(
            pair_surface.get("max_face_reprojection_error_px", raw.get("max_reprojection_error_px", 2.0))
        ),
        "post3d_enabled": bool(post3d.get("enabled", True)),
        "post3d_remove_rbm": bool(post3d.get("remove_rigid_body_motion", True)),
        "post3d_compute_face_measures": bool(post3d.get("compute_face_measures", True)),
        "post3d_compute_strain": bool(post3d.get("compute_strain", True)),
    }


def _write_recon3d_report(config: MDICConfig, report: dict[str, Any]) -> None:
    logs_dir = config.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "recon3d_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
