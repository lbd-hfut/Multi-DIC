from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackendUnavailableError, SfmPaths


class PycolmapBackend:
    name = "pycolmap"

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def run(self, paths: SfmPaths, image_names: list[str], report: dict) -> list[dict]:
        try:
            import pycolmap
        except ImportError as exc:
            raise BackendUnavailableError(
                "pycolmap is not available. Activate the project environment first: "
                "conda activate multi-dic"
            ) from exc

        self._prepare_workspace(paths)

        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = str(self.options.get("camera_model", "SIMPLE_RADIAL"))

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = bool(self.options.get("use_gpu", False))
        extraction_options.sift.max_num_features = int(self.options.get("max_features", 8192))
        extraction_options.sift.first_octave = int(self.options.get("first_octave", 0))
        if "max_image_size" in self.options:
            extraction_options.max_image_size = int(self.options["max_image_size"])
        if "num_threads" in self.options:
            extraction_options.num_threads = int(self.options["num_threads"])

        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = bool(self.options.get("use_gpu", False))
        matching_options.sift.cross_check = bool(self.options.get("cross_check", False))
        if "num_threads" in self.options:
            matching_options.num_threads = int(self.options["num_threads"])

        if "random_seed" in self.options:
            pycolmap.set_random_seed(int(self.options["random_seed"]))

        mapping_options = pycolmap.IncrementalPipelineOptions()
        mapping_options.image_names = image_names
        mapping_options.multiple_models = bool(self.options.get("multiple_models", True))
        mapping_options.max_num_models = int(self.options.get("max_num_models", mapping_options.max_num_models))
        mapping_options.min_model_size = min(int(self.options.get("min_model_size", 3)), len(image_names))
        mapping_options.min_num_matches = int(self.options.get("min_num_matches", mapping_options.min_num_matches))
        mapping_options.ba_global_max_refinements = int(
            self.options.get("ba_global_max_refinements", mapping_options.ba_global_max_refinements)
        )
        mapping_options.min_focal_length_ratio = float(
            self.options.get("min_focal_length_ratio", mapping_options.min_focal_length_ratio)
        )
        mapping_options.max_focal_length_ratio = float(
            self.options.get("max_focal_length_ratio", mapping_options.max_focal_length_ratio)
        )
        if "num_threads" in self.options:
            mapping_options.num_threads = int(self.options["num_threads"])
            mapping_options.mapper.num_threads = int(self.options["num_threads"])
        if "random_seed" in self.options:
            mapping_options.random_seed = int(self.options["random_seed"])
            mapping_options.mapper.random_seed = int(self.options["random_seed"])
            mapping_options.triangulation.random_seed = int(self.options["random_seed"])
        if "init_min_num_inliers" in self.options:
            mapping_options.mapper.init_min_num_inliers = int(self.options["init_min_num_inliers"])
        if "abs_pose_min_num_inliers" in self.options:
            mapping_options.mapper.abs_pose_min_num_inliers = int(self.options["abs_pose_min_num_inliers"])
        if "abs_pose_min_inlier_ratio" in self.options:
            mapping_options.mapper.abs_pose_min_inlier_ratio = float(self.options["abs_pose_min_inlier_ratio"])

        self._run_step(
            report,
            "extract_features",
            pycolmap.extract_features,
            paths.database_path,
            paths.image_root,
            image_names,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )
        self._run_step(
            report,
            "match_exhaustive",
            pycolmap.match_exhaustive,
            paths.database_path,
            matching_options=matching_options,
        )
        reconstructions = self._run_step(
            report,
            "incremental_mapping",
            pycolmap.incremental_mapping,
            paths.database_path,
            paths.image_root,
            paths.sparse_root,
            options=mapping_options,
        )
        summaries = _summarize_reconstructions(reconstructions, paths)
        best_model_id, best_reconstruction = _select_best_reconstruction(reconstructions)
        if best_reconstruction is not None:
            products = _export_ndef_sfm_products(
                best_reconstruction,
                paths,
                image_names,
                reference_camera=str(self.options.get("reference_camera", "cam_0")),
                max_reproj_error=float(self.options.get("max_reproj_error", 4.0)),
                dpi=int(self.options.get("dpi", 180)),
            )
            report["selected_model_id"] = int(best_model_id)
            report["products"] = products
            report["figures"] = {
                "sparse_scene": products.get("sparse_scene"),
                "camera_observations_3d": products.get("camera_observations_3d"),
                "camera_observations_2d": products.get("camera_observations_2d"),
            }
        return summaries

    def _prepare_workspace(self, paths: SfmPaths) -> None:
        paths.workspace.mkdir(parents=True, exist_ok=True)
        paths.sparse_root.mkdir(parents=True, exist_ok=True)
        if self.options.get("overwrite", True):
            if paths.database_path.exists():
                paths.database_path.unlink()
            if paths.sparse_root.exists():
                shutil.rmtree(paths.sparse_root)
            paths.sparse_root.mkdir(parents=True, exist_ok=True)

    def _run_step(self, report: dict, name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        step = {"name": name, "ok": False, "elapsed_seconds": None}
        report["steps"].append(step)
        result = func(*args, **kwargs)
        step["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        step["ok"] = True
        return result


def _summarize_reconstructions(reconstructions: Any, paths: SfmPaths) -> list[dict]:
    summaries: list[dict] = []
    for model_id, reconstruction in _iter_reconstructions(reconstructions):
        summaries.append(
            {
                "model_id": int(model_id),
                "path": str(paths.sparse_root / str(model_id)),
                "num_images": _maybe_call(reconstruction, "num_images"),
                "num_reg_images": _maybe_call(reconstruction, "num_reg_images"),
                "num_points3D": _maybe_call(reconstruction, "num_points3D"),
                "mean_reprojection_error": _maybe_call(reconstruction, "compute_mean_reprojection_error"),
            }
        )
    return summaries


def _iter_reconstructions(reconstructions: Any) -> list[tuple[int, Any]]:
    if isinstance(reconstructions, dict):
        return sorted((int(k), v) for k, v in reconstructions.items())
    if isinstance(reconstructions, (list, tuple)):
        return [(idx, reconstruction) for idx, reconstruction in enumerate(reconstructions)]
    try:
        return sorted((int(k), v) for k, v in dict(reconstructions).items())
    except Exception:
        return []


def _maybe_call(obj: Any, name: str) -> Any:
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    try:
        value = attr() if callable(attr) else attr
    except Exception:
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def _numeric(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _select_best_reconstruction(reconstructions: Any) -> tuple[int, Any | None]:
    items = _iter_reconstructions(reconstructions)
    if not items:
        return -1, None
    return max(
        items,
        key=lambda item: (
            _numeric(_maybe_call(item[1], "num_reg_images")),
            _numeric(_maybe_call(item[1], "num_points3D")),
        ),
    )


def _extract_camera_params(camera: Any) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(camera.params, dtype=np.float64)
    model_raw = str(camera.model)
    model = model_raw.split(".")[-1].upper() if "." in model_raw else model_raw.upper()

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = params[:9]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    else:
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    return K, dist


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ np.asarray(t).reshape(3)


def _get_cam_from_world(image: Any) -> tuple[np.ndarray, np.ndarray]:
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    return (
        np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64),
        np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3, 1),
    )


def _point_ids(reconstruction: Any) -> list[int]:
    ids = reconstruction.point3D_ids
    if callable(ids):
        ids = ids()
    return [int(pid) for pid in ids]


def _point3d(reconstruction: Any, point_id: int) -> Any:
    point_accessor = getattr(reconstruction, "point3D", None)
    if callable(point_accessor):
        return point_accessor(point_id)
    return reconstruction.points3D[point_id]


def _camera_model_name(camera: Any) -> str:
    model_raw = str(camera.model)
    return model_raw.split(".")[-1] if "." in model_raw else model_raw


def _to_reference_centroid_world(
    points_colmap: np.ndarray,
    R_colmap: list[np.ndarray],
    t_colmap: list[np.ndarray],
    ref_idx: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    centroid = points_colmap.mean(axis=0).astype(np.float64) if len(points_colmap) else np.zeros(3, dtype=np.float64)
    R_ref = R_colmap[ref_idx].astype(np.float64)
    points_world = (R_ref @ (points_colmap - centroid).T).T if len(points_colmap) else points_colmap

    R_world: list[np.ndarray] = []
    t_world: list[np.ndarray] = []
    for R, t in zip(R_colmap, t_colmap):
        t_vec = np.asarray(t).reshape(3)
        Rw = R @ R_ref.T
        tw = R @ centroid + t_vec
        R_world.append(Rw.astype(np.float64))
        t_world.append(tw.reshape(3, 1).astype(np.float64))
    return points_world.astype(np.float64), R_world, t_world, centroid, R_ref


def _extract_observations(
    reconstruction: Any,
    image_id_to_cam: dict[int, int],
    valid_point_ids: np.ndarray,
    points_world: np.ndarray,
    R_list: list[np.ndarray],
    t_list: list[np.ndarray],
    max_reproj_error: float,
) -> dict[str, np.ndarray]:
    pid_to_idx = {int(pid): idx for idx, pid in enumerate(valid_point_ids)}
    pid_to_xyz = {int(pid): points_world[idx] for idx, pid in enumerate(valid_point_ids)}

    point_indices: list[int] = []
    point_ids: list[int] = []
    cam_indices: list[int] = []
    uv: list[np.ndarray] = []
    depth: list[float] = []
    error: list[float] = []
    visibility = np.zeros((len(valid_point_ids), len(R_list)), dtype=bool)

    for image_id, image in reconstruction.images.items():
        image_id_int = int(image_id)
        if image_id_int not in image_id_to_cam:
            continue
        cam_idx = image_id_to_cam[image_id_int]
        R = R_list[cam_idx]
        t = t_list[cam_idx]

        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pid = int(p2d.point3D_id)
            if pid not in pid_to_idx:
                continue
            point = _point3d(reconstruction, pid)
            if float(point.error) > max_reproj_error:
                continue

            pidx = pid_to_idx[pid]
            xyz = pid_to_xyz[pid].reshape(3, 1)
            z = float((R @ xyz + t)[2, 0])
            if z <= 1e-8:
                continue

            point_indices.append(pidx)
            point_ids.append(pid)
            cam_indices.append(cam_idx)
            uv.append(np.asarray(p2d.xy, dtype=np.float64).reshape(2))
            depth.append(z)
            error.append(float(point.error))
            visibility[pidx, cam_idx] = True

    track_lengths = visibility.sum(axis=1).astype(np.int32)
    return {
        "point_indices": np.asarray(point_indices, dtype=np.int64),
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "cam_indices": np.asarray(cam_indices, dtype=np.int32),
        "uv": np.asarray(uv, dtype=np.float64).reshape(-1, 2),
        "depth": np.asarray(depth, dtype=np.float64),
        "reproj_error": np.asarray(error, dtype=np.float64),
        "visibility": visibility,
        "track_lengths": track_lengths,
    }


def _save_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")


def _export_ndef_sfm_products(
    reconstruction: Any,
    paths: SfmPaths,
    flat_image_names: list[str],
    reference_camera: str,
    max_reproj_error: float,
    dpi: int,
) -> dict[str, str]:
    from scipy.io import savemat

    cam_names = list(paths.camera_names)
    if reference_camera not in cam_names:
        raise ValueError(f"Reference camera {reference_camera!r} not in {cam_names}")
    ref_idx = cam_names.index(reference_camera)

    cam_image_ids: dict[str, list[int]] = {name: [] for name in cam_names}
    image_id_to_cam: dict[int, int] = {}
    for image_id, image in reconstruction.images.items():
        for idx, cam_name in enumerate(cam_names):
            if image.name.startswith(f"{cam_name}_"):
                cam_image_ids[cam_name].append(int(image_id))
                image_id_to_cam[int(image_id)] = idx
                break

    K_list: list[np.ndarray] = []
    dist_list: list[np.ndarray] = []
    R_colmap: list[np.ndarray] = []
    t_colmap: list[np.ndarray] = []
    camera_models: list[str] = []
    registered_image_names = [""] * len(cam_names)
    for idx, cam_name in enumerate(cam_names):
        ids = cam_image_ids[cam_name]
        if not ids:
            raise RuntimeError(f"Camera {cam_name} was not registered by COLMAP.")
        image = reconstruction.images[ids[0]]
        registered_image_names[idx] = image.name
        camera = reconstruction.cameras[image.camera_id]
        K, dist = _extract_camera_params(camera)
        R, t = _get_cam_from_world(image)
        K_list.append(K)
        dist_list.append(dist)
        R_colmap.append(R)
        t_colmap.append(t)
        camera_models.append(_camera_model_name(camera))

    point_ids: list[int] = []
    point_xyz: list[np.ndarray] = []
    point_errors: list[float] = []
    for point_id in _point_ids(reconstruction):
        point = _point3d(reconstruction, point_id)
        if float(point.error) <= max_reproj_error:
            point_ids.append(int(point_id))
            point_xyz.append(np.asarray(point.xyz, dtype=np.float64))
            point_errors.append(float(point.error))

    point_ids_arr = np.asarray(point_ids, dtype=np.int64)
    point_errors_arr = np.asarray(point_errors, dtype=np.float64)
    points_colmap = np.asarray(point_xyz, dtype=np.float64).reshape(-1, 3)
    points, R_list, t_list, centroid, R_ref = _to_reference_centroid_world(
        points_colmap, R_colmap, t_colmap, ref_idx
    )
    P_list = [K @ np.hstack((R, t.reshape(3, 1))) for K, R, t in zip(K_list, R_list, t_list)]
    observations = _extract_observations(
        reconstruction,
        image_id_to_cam,
        point_ids_arr,
        points,
        R_list,
        t_list,
        max_reproj_error,
    )

    K = np.stack(K_list, axis=0)
    dist = np.stack(dist_list, axis=0)
    R = np.stack(R_list, axis=0)
    t = np.stack([vec.reshape(3) for vec in t_list], axis=0)
    P = np.stack(P_list, axis=0)
    centers = np.stack([_camera_center(R_list[idx], t_list[idx]) for idx in range(len(cam_names))])
    image_paths = [str(path) for path in paths.source_image_paths]

    np.savez(
        paths.workspace / "cameras.npz",
        cam_names=np.array(cam_names),
        image_names=np.array(registered_image_names),
        image_paths=np.array(image_paths),
        camera_models=np.array(camera_models),
        K=K,
        dist=dist,
        R=R,
        t=t,
        P=P,
        camera_centers_world=centers,
        reference_camera=np.array(reference_camera),
        sparse_centroid_colmap=centroid,
        reference_rotation_colmap=R_ref,
    )
    np.savez(
        paths.workspace / "sparse_points.npz",
        points3D=points,
        point_ids=point_ids_arr,
        reproj_error=point_errors_arr,
        visibility=observations["visibility"],
        track_lengths=observations["track_lengths"],
    )
    np.savez(
        paths.workspace / "observations.npz",
        cam_names=np.array(cam_names),
        point_indices=observations["point_indices"],
        point_ids=observations["point_ids"],
        cam_indices=observations["cam_indices"],
        uv=observations["uv"],
        depth=observations["depth"],
        reproj_error=observations["reproj_error"],
    )
    savemat(
        paths.workspace / "cameras.mat",
        {
            "num_cameras": len(cam_names),
            "K_list": K,
            "dist_list": dist,
            "cam_from_world_R": R,
            "cam_from_world_t": t,
            "P_list": P,
            "camera_centers_world": centers,
            "camera_models": np.array(camera_models, dtype=object),
            "cam_names": np.array(cam_names, dtype=object),
            "image_names": np.array(registered_image_names, dtype=object),
            "reference_camera": reference_camera,
            "sparse_centroid_colmap": centroid,
            "reference_rotation_colmap": R_ref,
            "num_registered_images": _numeric(_maybe_call(reconstruction, "num_reg_images")),
        },
    )
    savemat(
        paths.workspace / "points3D.mat",
        {
            "points3D": points,
            "point_ids": point_ids_arr,
            "reproj_error": point_errors_arr,
            "visibility": observations["visibility"].astype(np.uint8),
            "track_lengths": observations["track_lengths"],
            "num_points": len(points),
        },
    )

    _write_camera_text_products(
        paths.workspace,
        cam_names,
        registered_image_names,
        image_paths,
        camera_models,
        K,
        dist,
        R,
        t,
        P,
        centers,
        reference_camera,
        centroid,
        _numeric(_maybe_call(reconstruction, "num_reg_images")),
        points,
        observations,
    )
    _save_ply(paths.workspace / "sparse_points.ply", points)
    _save_ply(paths.workspace / "camera_centers.ply", centers)
    figure_paths = _visualize_products(paths.workspace, cam_names, image_paths, R_list, t_list, points, observations, dpi)

    return {
        "cameras_npz": str(paths.workspace / "cameras.npz"),
        "cameras_mat": str(paths.workspace / "cameras.mat"),
        "cameras_json": str(paths.workspace / "cameras.json"),
        "cameras_txt": str(paths.workspace / "cameras.txt"),
        "sparse_points_npz": str(paths.workspace / "sparse_points.npz"),
        "points3D_mat": str(paths.workspace / "points3D.mat"),
        "observations_npz": str(paths.workspace / "observations.npz"),
        "sparse_points_ply": str(paths.workspace / "sparse_points.ply"),
        "camera_centers_ply": str(paths.workspace / "camera_centers.ply"),
        **figure_paths,
    }


def _write_camera_text_products(
    out_dir: Path,
    cam_names: list[str],
    image_names: list[str],
    image_paths: list[str],
    camera_models: list[str],
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    P: np.ndarray,
    centers: np.ndarray,
    reference_camera: str,
    centroid_colmap: np.ndarray,
    num_registered: int,
    points: np.ndarray,
    observations: dict[str, np.ndarray],
) -> None:
    data = {
        "coordinate_system": {
            "origin": "centroid of retained sparse COLMAP points",
            "axes": f"parallel to {reference_camera} camera axes",
            "note": "COLMAP scale is still arbitrary until an external scale constraint is applied.",
            "sparse_centroid_colmap": centroid_colmap.tolist(),
        },
        "num_cameras": len(cam_names),
        "num_registered_images": int(num_registered),
        "cameras": [],
    }
    for idx, name in enumerate(cam_names):
        data["cameras"].append(
            {
                "index": idx,
                "name": name,
                "image_name": image_names[idx],
                "image_path": image_paths[idx],
                "model": camera_models[idx],
                "K": K[idx].tolist(),
                "distortion": dist[idx].tolist(),
                "cam_from_world_R": R[idx].tolist(),
                "cam_from_world_t": t[idx].tolist(),
                "world_position": centers[idx].tolist(),
                "projection_matrix": P[idx].tolist(),
            }
        )
    with (out_dir / "cameras.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "SfM camera parameters",
        f"Output frame: origin=sparse centroid, axes={reference_camera}",
        f"Registered images: {num_registered}",
        f"Sparse points: {len(points)}",
        f"Track observations: {len(observations['uv'])}",
        "",
    ]
    for idx, name in enumerate(cam_names):
        center = centers[idx]
        lines.extend(
            [
                f"{idx:02d} {name}",
                f"  image: {image_names[idx]}",
                f"  model: {camera_models[idx]}",
                f"  f_px: ({K[idx, 0, 0]:.9g}, {K[idx, 1, 1]:.9g})",
                f"  principal_point_px: ({K[idx, 0, 2]:.9g}, {K[idx, 1, 2]:.9g})",
                f"  world_position: ({center[0]:.9g}, {center[1]:.9g}, {center[2]:.9g})",
                "",
            ]
        )
    with (out_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _axis_limits(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    all_points = points if len(points) else centers
    all_points = np.vstack([all_points, centers])
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    mid = 0.5 * (lo + hi)
    span = max(float(np.max(hi - lo)), 1.0)
    half = 0.55 * span
    return np.stack([mid - half, mid + half], axis=0)


def _apply_limits(ax: Any, limits: np.ndarray) -> None:
    ax.set_xlim(limits[0, 0], limits[1, 0])
    ax.set_ylim(limits[0, 1], limits[1, 1])
    ax.set_zlim(limits[0, 2], limits[1, 2])


def _visualize_products(
    out_dir: Path,
    cam_names: list[str],
    image_paths: list[str],
    R_list: list[np.ndarray],
    t_list: list[np.ndarray],
    points: np.ndarray,
    observations: dict[str, np.ndarray],
    dpi: int,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.stack([_camera_center(R_list[idx], t_list[idx]) for idx in range(len(cam_names))])
    limits = _axis_limits(points, centers)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        count = min(len(points), 30000)
        indices = np.random.RandomState(0).choice(len(points), count, replace=False)
        sample = points[indices]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.0, c="0.25", alpha=0.55)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=45, marker="^")
    scale = 0.08 * max(limits[1] - limits[0])
    for idx, center in enumerate(centers):
        ax.text(center[0], center[1], center[2], cam_names[idx], fontsize=7)
        axes = scale * R_list[idx].T
        for axis, color in zip(axes.T, ["r", "g", "b"]):
            ax.quiver(center[0], center[1], center[2], axis[0], axis[1], axis[2], color=color, linewidth=1)
    _apply_limits(ax, limits)
    ax.set_title("Sparse points and camera poses")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    sparse_scene = out_dir / "sparse_scene.png"
    fig.savefig(sparse_scene, dpi=dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12))
    for idx, name in enumerate(cam_names):
        ax = fig.add_subplot(3, 4, idx + 1, projection="3d")
        mask = observations["cam_indices"] == idx
        point_indices = observations["point_indices"][mask]
        observed_points = points[point_indices] if len(point_indices) else np.zeros((0, 3))
        if len(observed_points):
            ax.scatter(observed_points[:, 0], observed_points[:, 1], observed_points[:, 2], s=2, c="tab:red", alpha=0.75)
        _apply_limits(ax, limits)
        ax.set_title(f"{name}: {len(observed_points)} obs", fontsize=9)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    observations_3d = out_dir / "camera_observations_3d.png"
    fig.savefig(observations_3d, dpi=dpi)
    plt.close(fig)

    import matplotlib.image as mpimg

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes_flat = axes.ravel()
    for idx, name in enumerate(cam_names):
        ax = axes_flat[idx]
        image = mpimg.imread(image_paths[idx])
        if image.ndim == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")
        mask = observations["cam_indices"] == idx
        uv = observations["uv"][mask]
        if len(uv):
            ax.scatter(uv[:, 0], uv[:, 1], s=18, c="red", marker="o", linewidths=0)
        ax.set_title(f"{name}: {len(uv)} obs", fontsize=10)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
    for idx in range(len(cam_names), len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.tight_layout()
    observations_2d = out_dir / "camera_observations_2d.png"
    fig.savefig(observations_2d, dpi=dpi)
    plt.close(fig)

    return {
        "sparse_scene": str(sparse_scene),
        "camera_observations_3d": str(observations_3d),
        "camera_observations_2d": str(observations_2d),
    }
