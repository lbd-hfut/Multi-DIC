from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from multidic.config import (
    DataConfig,
    MDICConfig,
    OutputConfig,
    ProjectConfig,
    load_config as _load_config,
)

STEP_NAMES = ("validate", "sfm", "scale", "mask", "dic2d", "recon3d", "visualize3d")

_DEFAULT_RAW: dict[str, Any] = {
    "project": {
        "name": "PyMultiDIC",
        "case_root": "",
        "output_root": "results",
    },
    "data": {
        "speckle_dir": "images",
        "calibration_dir": "calibrate_images",
        "camera_glob": "cam_*",
        "reference_frame": "001.bmp",
        "deformed_frames": ["002.bmp"],
        "image_extensions": [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
    },
    "scale_correction": {
        "checkerboard_meta": "calibrate_images/chessboard_meta.json",
        "image_dir": "calibrate_images",
        "square_size": 10.0,
        "square_size_unit": "mm",
        "pair_selection": "all_pairs",
        "scale_stat": "trimmed_mean",
        "trim_fraction": 0.20,
        "max_pair_edge_cv": 0.08,
        "max_edge_cv_warning": 0.05,
        "corner_order_edge_cv_weight": 1.0,
        "subpix_window": 11,
        "min_common_corners": 12,
        "max_reprojection_error_px": 3.0,
        "save_overlays": True,
        "output_dir": "scale",
    },
    "mask": {
        "user_mask_dir": "masks",
        "use_user_mask_if_present": True,
        "output_dir": "masks",
        "external_threshold": 127,
        "outlier_k": 6,
        "outlier_knn_scale": 4.0,
        "component_radius_scale": 8.0,
        "edge_scale": 8.0,
        "radius_scale": 6.0,
        "min_hole_area": 500,
        "tiny_hole_fill_area": 3000,
        "speckle_std_ratio": 0.35,
        "speckle_lap_ratio": 0.35,
        "speckle_grad_ratio": 0.35,
        "min_speckle_std": 6.0,
        "min_speckle_lap": 3.0,
        "overlay_alpha": 0.45,
        "dpi": 180,
    },
    "dic2d": {
        "engine": "ncorr",
        "analysis_type": "regular",
        "subset_radius": 20,
        "subset_spacing": 5,
        "seed_search_radius": 50,
        "rg_search_radius": 1,
        "cutoff_diffnorm": 1.0e-6,
        "cutoff_iteration": 50,
        "num_threads": 1,
        "subset_truncation": False,
        "step_analysis": {"enabled": False, "type": "seed", "auto": True, "step": 5},
        "roi": {
            "source": "auto_or_user",
            "user_roi_mode": "last_image",
            "mask_output_dir": "masks/mask",
            "external_threshold": 127,
            "min_region_area": 2000,
        },
        "seed": {"source": "colmap_observations", "selection": "first_inside_roi"},
        "format": {
            "units_per_pixel": 1.0,
            "units": "pixels",
            "cutoff_corrcoef": 0.6,
            "lenscoef": 0.0,
        },
        "strain": {"enabled": False},
    },
    "colmap": {
        "backend": "native_colmap",
        "executable": "colmap",
        "allow_external_executable": False,
        "workspace": "colmap",
        "reference_camera": "cam_0",
        "camera_model": "SIMPLE_RADIAL",
        "matcher": "ring",
        "matching_window": 2,
        "wrap_matching": True,
        "use_gpu": False,
        "overwrite": True,
        "max_features": 8192,
        "first_octave": -1,
        "cross_check": False,
        "min_num_matches": 8,
        "multiple_models": False,
        "min_model_size": 12,
        "init_min_num_inliers": 50,
        "init_max_error": 4.0,
        "abs_pose_min_num_inliers": 15,
        "abs_pose_min_inlier_ratio": 0.05,
        "abs_pose_max_error": 12.0,
        "filter_max_reproj_error": 4.0,
        "abs_pose_refine_focal_length": True,
        "abs_pose_refine_extra_params": True,
        "ba_global_max_refinements": 5,
        "max_reproj_error": 4.0,
        "dpi": 180,
        "min_focal_length_ratio": 0.1,
        "max_focal_length_ratio": 10.0,
        "random_seed": 1,
        "num_threads": 1,
    },
    "recon3d": {
        "backend": "native",
        "output_dir": "recon3d",
        "input_dic2d_dir": "dic2d",
        "scale_dir": "scale",
        "min_views": 2,
        "min_corrcoef": 0.6,
        "max_reprojection_error_px": 2.0,
        "use_scale_correction": True,
        "outlier_filter": {
            "enabled": True,
            "min_points": 20,
            "position_mad_z": 8.0,
            "displacement_mad_z": 8.0,
            "max_position_radius_world": 0.0,
            "max_displacement_norm_world": 0.0,
        },
        "pairs": {
            "mode": "auto_spatial",
            "wrap": True,
            "manual": [],
            "auto_circularity_threshold": 0.45,
            "auto_wrap_distance_ratio": 1.8,
            "auto_wrap_min_shared_ratio": 0.35,
            "auto_max_neighbor_distance_ratio": 2.0,
            "auto_min_shared_tracks": 20,
        },
        "pair_surface": {
            "enabled": True,
            "triangulation_domain": "reference_uv",
            "max_edge_px": 80.0,
            "min_face_corrcoef": 0.6,
            "max_face_reprojection_error_px": 2.0,
        },
        "post3d": {
            "enabled": True,
            "remove_rigid_body_motion": True,
            "compute_face_measures": True,
            "compute_strain": True,
        },
        "qc": {"enabled": True, "plots": True, "vector_scale": 1.0},
        "export": {"npz": True, "ply": True, "png": True, "qc_ply": True},
    },
    "visualization": {
        "output_dir": "figures",
        "dpi": 180,
        "point_size": 7.0,
        "surface_alpha": 0.28,
        "max_points": 60000,
        "surface_theta_samples": 260,
        "surface_y_samples": 190,
        "view_elev": 22.0,
        "view_azim": -58.0,
    },
    "output": {
        "subdirectories": ["logs", "sfm", "scale", "masks", "dic2d", "recon3d", "figures"],
    },
}

_SECTION_NAMES = {
    "project",
    "data",
    "scale_correction",
    "sfm2world",
    "mask",
    "dic2d",
    "colmap",
    "recon3d",
    "visualization",
    "output",
}

_FLAT_OVERRIDE_PATHS: dict[str, tuple[str, ...]] = {
    "checkerboard_meta": ("scale_correction", "checkerboard_meta"),
    "square_size": ("scale_correction", "square_size"),
    "square_size_unit": ("scale_correction", "square_size_unit"),
    "subset_radius": ("dic2d", "subset_radius"),
    "subset_spacing": ("dic2d", "subset_spacing"),
    "seed_search_radius": ("dic2d", "seed_search_radius"),
    "rg_search_radius": ("dic2d", "rg_search_radius"),
    "cutoff_diffnorm": ("dic2d", "cutoff_diffnorm"),
    "cutoff_iteration": ("dic2d", "cutoff_iteration"),
    "num_threads": ("dic2d", "num_threads"),
    "subset_truncation": ("dic2d", "subset_truncation"),
    "units_per_pixel": ("dic2d", "format", "units_per_pixel"),
    "cutoff_corrcoef": ("dic2d", "format", "cutoff_corrcoef"),
    "lenscoef": ("dic2d", "format", "lenscoef"),
    "colmap_backend": ("colmap", "backend"),
    "colmap_executable": ("colmap", "executable"),
    "colmap_workspace": ("colmap", "workspace"),
    "reference_camera": ("colmap", "reference_camera"),
    "camera_model": ("colmap", "camera_model"),
    "matcher": ("colmap", "matcher"),
    "matching_window": ("colmap", "matching_window"),
    "wrap_matching": ("colmap", "wrap_matching"),
    "use_gpu": ("colmap", "use_gpu"),
    "max_features": ("colmap", "max_features"),
    "first_octave": ("colmap", "first_octave"),
    "cross_check": ("colmap", "cross_check"),
    "min_num_matches": ("colmap", "min_num_matches"),
    "multiple_models": ("colmap", "multiple_models"),
    "min_model_size": ("colmap", "min_model_size"),
    "init_min_num_inliers": ("colmap", "init_min_num_inliers"),
    "init_max_error": ("colmap", "init_max_error"),
    "abs_pose_min_num_inliers": ("colmap", "abs_pose_min_num_inliers"),
    "abs_pose_min_inlier_ratio": ("colmap", "abs_pose_min_inlier_ratio"),
    "abs_pose_max_error": ("colmap", "abs_pose_max_error"),
    "filter_max_reproj_error": ("colmap", "filter_max_reproj_error"),
    "abs_pose_refine_focal_length": ("colmap", "abs_pose_refine_focal_length"),
    "abs_pose_refine_extra_params": ("colmap", "abs_pose_refine_extra_params"),
    "ba_global_max_refinements": ("colmap", "ba_global_max_refinements"),
    "colmap_random_seed": ("colmap", "random_seed"),
    "recon_backend": ("recon3d", "backend"),
    "min_views": ("recon3d", "min_views"),
    "min_corrcoef": ("recon3d", "min_corrcoef"),
    "max_reprojection_error_px": ("recon3d", "max_reprojection_error_px"),
    "outlier_filter_enabled": ("recon3d", "outlier_filter", "enabled"),
    "position_mad_z": ("recon3d", "outlier_filter", "position_mad_z"),
    "displacement_mad_z": ("recon3d", "outlier_filter", "displacement_mad_z"),
    "visualization_output_dir": ("visualization", "output_dir"),
    "view_elev": ("visualization", "view_elev"),
    "view_azim": ("visualization", "view_azim"),
    "dpi": ("visualization", "dpi"),
}


def load_config(config_path: str | Path, workspace_root: str | Path | None = None) -> MDICConfig:
    """Load a Multi-DIC YAML config file."""

    workspace = Path(workspace_root) if workspace_root is not None else None
    return _load_config(Path(config_path), workspace_root=workspace)


def build_config(
    config: MDICConfig | None = None,
    *,
    case_root: str | Path | None = None,
    project_name: str = "PyMultiDIC",
    output_root: str | Path | None = None,
    speckle_dir: str | Path = "images",
    calibration_dir: str | Path = "calibrate_images",
    camera_glob: str = "cam_*",
    reference_frame: str = "001.bmp",
    deformed_frames: list[str] | tuple[str, ...] | None = None,
    image_extensions: list[str] | tuple[str, ...] | None = None,
    workspace_root: str | Path | None = None,
    raw_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> MDICConfig:
    """Return an MDICConfig from either a config object or direct API inputs.

    If ``config`` is provided, its raw settings are used as the base and
    explicit overrides are applied. If ``config`` is ``None``, ``case_root`` is
    required and the remaining inputs override the default workflow parameters.
    """

    if config is not None:
        if output_root is not None:
            overrides.setdefault("project", {})["output_root"] = str(output_root)
        if not raw_overrides and not overrides:
            return config
        raw = deepcopy(config.raw)
        if raw_overrides:
            _deep_update(raw, raw_overrides)
        _apply_keyword_overrides(raw, overrides)
        return _config_from_raw(raw, config.workspace_root)
    if case_root is None:
        raise ValueError("case_root is required when config is None.")

    workspace = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    raw = deepcopy(_DEFAULT_RAW)
    raw["project"].update(
        {
            "name": project_name,
            "case_root": str(case_root),
            "output_root": str(output_root or "results"),
        }
    )
    raw["data"].update(
        {
            "speckle_dir": str(speckle_dir),
            "calibration_dir": str(calibration_dir),
            "camera_glob": camera_glob,
            "reference_frame": reference_frame,
            "deformed_frames": list(deformed_frames) if deformed_frames is not None else ["002.bmp"],
        }
    )
    if image_extensions is not None:
        raw["data"]["image_extensions"] = list(image_extensions)
    raw["scale_correction"]["image_dir"] = str(calibration_dir)
    raw["scale_correction"]["checkerboard_meta"] = str(Path(calibration_dir) / "chessboard_meta.json")

    if raw_overrides:
        _deep_update(raw, raw_overrides)
    _apply_keyword_overrides(raw, overrides)
    return _config_from_raw(raw, workspace)


def validate_project(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    """Validate project inputs and create configured output directories."""

    from multidic.validate import validate_case

    return validate_case(build_config(config, **kwargs)).to_dict()


def run_validate(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`validate_project`."""

    return validate_project(config, **kwargs)


def run_sfm(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    from multidic.sfm import run_sfm as _run_sfm

    return _run_sfm(build_config(config, **kwargs))


def run_scale(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    from multidic.scale import run_scale as _run_scale

    return _run_scale(build_config(config, **kwargs))


def run_mask(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    from multidic.mask import run_mask as _run_mask

    return _run_mask(build_config(config, **kwargs))


def run_dic2d(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    from multidic.dic2d import run_dic2d as _run_dic2d

    return _run_dic2d(build_config(config, **kwargs))


def run_recon3d(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    from multidic.recon3d import run_recon3d as _run_recon3d

    return _run_recon3d(build_config(config, **kwargs))


def run_visualize3d(config: MDICConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    from multidic.visualization import run_visualize3d as _run_visualize3d

    return _run_visualize3d(build_config(config, **kwargs))


def run_step(config: MDICConfig | str | None = None, step: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Run one workflow step by name."""

    if isinstance(config, str) and step is None:
        step = config
        config = None
    if step is None:
        raise ValueError("step is required.")
    step_name = step.lower()
    runners: dict[str, Callable[..., dict[str, Any]]] = {
        "validate": run_validate,
        "sfm": run_sfm,
        "scale": run_scale,
        "mask": run_mask,
        "dic2d": run_dic2d,
        "recon3d": run_recon3d,
        "visualize3d": run_visualize3d,
    }
    try:
        resolved_config = config if isinstance(config, MDICConfig) else None
        return runners[step_name](resolved_config, **kwargs)
    except KeyError as exc:
        expected = ", ".join(STEP_NAMES)
        raise ValueError(f"Unknown step {step!r}; expected one of: {expected}") from exc


def run_pipeline(
    config: MDICConfig | None = None,
    steps: list[str] | tuple[str, ...] | None = None,
    *,
    stop_on_error: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a sequence of workflow steps and return a report keyed by step name."""

    resolved_config = build_config(config, **kwargs)
    selected_steps = tuple(steps) if steps is not None else STEP_NAMES[:-1]
    reports: dict[str, Any] = {}
    ok = True
    for step in selected_steps:
        report = run_step(resolved_config, step)
        reports[step] = report
        step_ok = bool(report.get("ok", False))
        ok = ok and step_ok
        if stop_on_error and not step_ok:
            break
    return {
        "ok": ok,
        "steps": list(selected_steps),
        "completed_steps": list(reports.keys()),
        "reports": reports,
    }


def _config_from_raw(raw: dict[str, Any], workspace: Path) -> MDICConfig:
    project = raw["project"]
    data = raw["data"]
    output = raw["output"]
    case_root = _resolve_workspace_path(str(project["case_root"]), workspace)
    output_root = Path(str(project["output_root"]))
    if output_root.is_absolute() or any(part == ".." for part in output_root.parts):
        raise ValueError("output_root must be a relative path inside case_root.")
    image_extensions = tuple(
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in data["image_extensions"]
    )
    return MDICConfig(
        path=workspace / "<direct-api>",
        workspace_root=workspace,
        project=ProjectConfig(
            name=str(project["name"]),
            case_root=case_root,
            output_root=output_root,
        ),
        data=DataConfig(
            speckle_dir=str(data["speckle_dir"]),
            calibration_dir=str(data["calibration_dir"]),
            camera_glob=str(data["camera_glob"]),
            reference_frame=str(data["reference_frame"]),
            deformed_frames=tuple(str(frame) for frame in data["deformed_frames"]),
            image_extensions=image_extensions,
        ),
        output=OutputConfig(subdirectories=tuple(str(item) for item in output["subdirectories"])),
        raw=raw,
    )


def _resolve_workspace_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else workspace / path).resolve()


def _apply_keyword_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    unknown = []
    for key, value in overrides.items():
        if value is None:
            continue
        if key in _SECTION_NAMES:
            if not isinstance(value, dict):
                raise ValueError(f"{key} override must be a dict.")
            _deep_update(raw.setdefault(key, {}), value)
        elif key in _FLAT_OVERRIDE_PATHS:
            _set_nested(raw, _FLAT_OVERRIDE_PATHS[key], value)
        else:
            unknown.append(key)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown direct API parameter(s): {names}. Use raw_overrides={{...}} for nested config values.")


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value
