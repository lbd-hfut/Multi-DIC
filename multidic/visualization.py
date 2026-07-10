from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import Normalize

from .config import MDICConfig


def run_visualize3d(config: MDICConfig) -> dict[str, Any]:
    cfg = _visualize_config(config)
    figures_dir = cfg["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "project": config.project.name,
        "frame": cfg["frame_stem"],
        "figures_dir": str(figures_dir),
        "outputs": {},
        "errors": [],
        "warnings": [],
    }
    try:
        recon = _load_npz(cfg["recon_npz"])
        valid = np.asarray(recon["valid"], dtype=bool)
        points = np.asarray(recon["points_ref_world"], dtype=np.float64)
        displacement = np.asarray(recon["displacement_world"], dtype=np.float64)
        disp_norm = np.linalg.norm(displacement, axis=1)
        pair_surfaces = _load_pair_surfaces(cfg["pair_dir"], cfg["recon_report"], cfg["frame_stem"])
        if not np.any(valid):
            raise ValueError(f"No valid 3D points in {cfg['recon_npz']}")
        surface_points, surface_displacement = _collect_surface_samples(pair_surfaces, points, displacement, valid)
        surface = _interpolate_cylinder_surface_fields(surface_points, surface_displacement, cfg)

        outputs: dict[str, str] = {}
        outputs["initial_shape_points"] = str(
            _plot_initial_shape_points(
                figures_dir / f"{cfg['frame_stem']}_initial_shape_points.png",
                surface_points,
                np.ones(len(surface_points), dtype=bool),
                cfg,
            )
        )
        outputs["initial_shape_surface"] = str(
            _plot_initial_shape_surface(
                figures_dir / f"{cfg['frame_stem']}_initial_shape_surface.png",
                surface,
                cfg,
            )
        )
        outputs["surface_fields"] = str(
            _plot_surface_fields(
                figures_dir / f"{cfg['frame_stem']}_surface_fields.png",
                surface,
                cfg,
            )
        )
        outputs["displacement_components"] = str(
            _plot_displacement_components_scatter(
                figures_dir / f"{cfg['frame_stem']}_displacement_components.png",
                surface_points,
                surface_displacement,
                np.ones(len(surface_points), dtype=bool),
                cfg,
            )
        )
        if not pair_surfaces:
            report["warnings"].append(f"No pair surface npz files found in {cfg['pair_dir']}")
        else:
            surface_cloud_dir = figures_dir / "surface_clouds"
            outputs.update(
                _plot_pair_surface_cloud_maps(
                    surface_cloud_dir,
                    cfg["frame_stem"],
                    pair_surfaces,
                    cfg,
                )
            )
        outputs["reference_points_ply"] = str(
            _write_reference_points_ply(
                figures_dir / f"{cfg['frame_stem']}_reference_points.ply",
                surface_points,
                np.ones(len(surface_points), dtype=bool),
            )
        )
        outputs["surface_fields_npz"] = str(
            _write_surface_fields_npz(figures_dir / f"{cfg['frame_stem']}_surface_fields.npz", surface)
        )
        report["outputs"] = outputs
        report["stats"] = {
            "num_points_total": int(len(valid)),
            "num_points_valid": int(np.count_nonzero(valid)),
            "num_pair_surfaces": int(len(pair_surfaces)),
            "num_surface_samples": int(len(surface_points)),
            "num_surface_cloud_faces": int(_count_valid_pair_surface_faces(pair_surfaces)),
            "displacement_norm": _array_stats(disp_norm[valid]),
        }
        report["ok"] = True
    except Exception as exc:
        report["errors"].append(f"visualize3d failed: {type(exc).__name__}: {exc}")

    report_path = figures_dir / f"{cfg['frame_stem']}_visualize3d_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    report["report_json"] = str(report_path)
    return report


def _visualize_config(config: MDICConfig) -> dict[str, Any]:
    raw = config.raw.get("visualization", {})
    if not isinstance(raw, dict):
        raw = {}
    recon_raw = config.raw.get("recon3d", {})
    if not isinstance(recon_raw, dict):
        recon_raw = {}
    frame = str(raw.get("frame", Path(config.data.deformed_frames[0]).stem if config.data.deformed_frames else "002"))
    frame_stem = Path(frame).stem
    recon_dir = config.result_root / str(recon_raw.get("output_dir", raw.get("recon3d_dir", "recon3d")))
    return {
        "frame_stem": frame_stem,
        "recon_npz": recon_dir / f"recon3d_{frame_stem}.npz",
        "pair_dir": recon_dir / "pairs" / frame_stem,
        "recon_report": config.result_root / "logs" / "recon3d_report.json",
        "figures_dir": config.result_root / str(raw.get("output_dir", "figures")),
        "dpi": int(raw.get("dpi", 180)),
        "point_size": float(raw.get("point_size", 7.0)),
        "surface_alpha": float(raw.get("surface_alpha", 0.28)),
        "max_points": int(raw.get("max_points", 60000)),
        "surface_theta_samples": int(raw.get("surface_theta_samples", 260)),
        "surface_y_samples": int(raw.get("surface_y_samples", 190)),
        "view_elev": float(raw.get("view_elev", 22.0)),
        "view_azim": float(raw.get("view_azim", -58.0)),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_pair_surfaces(pair_dir: Path, recon_report: Path, frame_stem: str) -> list[dict[str, np.ndarray]]:
    paths = _pair_surface_paths_from_report(recon_report, frame_stem)
    if not paths:
        if not pair_dir.exists():
            return []
        paths = sorted(pair_dir.glob("pair_*_*.npz"))
    surfaces = []
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        if {"points_ref_world", "faces", "valid_faces", "displacement_world"}.issubset(payload):
            payload["_path"] = np.asarray(str(path))
            surfaces.append(payload)
    return surfaces


def _collect_surface_samples(
    pair_surfaces: list[dict[str, np.ndarray]],
    points: np.ndarray,
    displacement: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks: list[np.ndarray] = []
    disp_chunks: list[np.ndarray] = []
    for surface in pair_surfaces:
        pts = np.asarray(surface["points_ref_world"], dtype=np.float64)
        disp = np.asarray(surface["displacement_world"], dtype=np.float64)
        valid_points = np.asarray(surface["valid_points"], dtype=bool)
        faces = np.asarray(surface["faces"], dtype=np.int32)
        valid_faces = np.asarray(surface["valid_faces"], dtype=bool)
        used_faces = faces[valid_faces]
        if len(used_faces):
            used = np.zeros(len(pts), dtype=bool)
            used[np.unique(used_faces.reshape(-1))] = True
            valid_points &= used
        finite = valid_points & np.all(np.isfinite(pts), axis=1) & np.all(np.isfinite(disp), axis=1)
        if np.any(finite):
            point_chunks.append(pts[finite])
            disp_chunks.append(disp[finite])
    if point_chunks:
        return np.concatenate(point_chunks, axis=0), np.concatenate(disp_chunks, axis=0)
    finite = valid & np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(displacement), axis=1)
    return points[finite], displacement[finite]


def _pair_surface_paths_from_report(recon_report: Path, frame_stem: str) -> list[Path]:
    if not recon_report.exists():
        return []
    try:
        with recon_report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception:
        return []
    paths: list[Path] = []
    for frame in report.get("frames", []):
        if Path(str(frame.get("frame", ""))).stem != frame_stem:
            continue
        for item in frame.get("pair_surfaces", []):
            output = item.get("output_npz")
            if output:
                path = _normalize_report_path(str(output))
                if path.exists():
                    paths.append(path)
    return paths


def _normalize_report_path(path_text: str) -> Path:
    path = Path(path_text)
    text = path_text.replace("\\", "/")
    if not path.exists() and text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        return Path(f"{drive}:/{text[7:]}")
    return path


def _plot_initial_shape_points(path: Path, points: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = points[valid]
    fig = plt.figure(figsize=(8.2, 7.0), dpi=int(cfg["dpi"]))
    ax = fig.add_subplot(111, projection="3d")
    color_values = pts[:, 2] if len(pts) else np.zeros(0, dtype=np.float64)
    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color_values, s=float(cfg["point_size"]), cmap="viridis", linewidths=0)
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label="Z")
    ax.set_title("Initial 3D shape")
    _style_3d_axis(ax, pts, cfg)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_initial_pair_surfaces(path: Path, pair_surfaces: list[dict[str, np.ndarray]], cfg: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.2, 7.4), dpi=int(cfg["dpi"]))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    for idx, surface in enumerate(pair_surfaces):
        pts = np.asarray(surface["points_ref_world"], dtype=np.float64)
        faces = np.asarray(surface["faces"], dtype=np.int32)
        valid_faces = np.asarray(surface["valid_faces"], dtype=bool)
        used_faces = faces[valid_faces]
        if len(pts) == 0 or len(used_faces) == 0:
            continue
        all_points.append(pts[np.unique(used_faces.reshape(-1))])
        ax.plot_trisurf(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            triangles=used_faces,
            color=plt.cm.tab20(idx % 20),
            alpha=float(cfg["surface_alpha"]),
            linewidth=0.08,
            edgecolor="#4a5568",
            shade=True,
        )
    merged = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float64)
    ax.set_title("Initial pair surfaces")
    _style_3d_axis(ax, merged, cfg)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_initial_shape_surface(path: Path, surface: dict[str, np.ndarray], cfg: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.2, 7.4), dpi=int(cfg["dpi"]))
    ax = fig.add_subplot(111, projection="3d")
    value = surface["radius"]
    norm = Normalize(*_field_color_limits(value, signed=False))
    surf = ax.plot_surface(
        surface["x"],
        surface["y"],
        surface["z"],
        facecolors=plt.get_cmap("viridis")(norm(value)),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=0.96,
    )
    surf.set_edgecolor("none")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(value)
    ax.set_title("Interpolated initial morphology")
    _style_surface_axis(ax, surface, cfg)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("radius")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_surface_fields(path: Path, surface: dict[str, np.ndarray], cfg: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [surface["radius"], surface["magnitude"], surface["ux"], surface["uy"], surface["uz"]]
    titles = ["Initial morphology", "Total displacement", "U displacement", "V displacement", "W displacement"]
    labels = ["radius", "|U|", "U", "V", "W"]
    cmaps = ["viridis", "viridis", "coolwarm", "coolwarm", "coolwarm"]
    fig = plt.figure(figsize=(28, 5.6), dpi=int(cfg["dpi"]))
    for idx, (value, title, label, cmap) in enumerate(zip(values, titles, labels, cmaps), start=1):
        ax = fig.add_subplot(1, 5, idx, projection="3d")
        norm = Normalize(*_field_color_limits(value, signed=idx >= 3))
        surf = ax.plot_surface(
            surface["x"],
            surface["y"],
            surface["z"],
            facecolors=plt.get_cmap(cmap)(norm(value)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.96,
        )
        surf.set_edgecolor("none")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(value)
        ax.set_title(title)
        _style_surface_axis(ax, surface, cfg)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.12, fraction=0.030)
        cbar.set_label(label)
    fig.suptitle("Interpolated 3D surface morphology and displacement fields")
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.08, top=0.84, wspace=0.42)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_displacement_components_scatter(
    path: Path,
    points: np.ndarray,
    displacement: np.ndarray,
    valid: np.ndarray,
    cfg: dict[str, Any],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = points[valid]
    disp = displacement[valid]
    if len(pts) > int(cfg["max_points"]):
        rng = np.random.default_rng(20260707)
        order = rng.choice(len(pts), size=int(cfg["max_points"]), replace=False)
        pts = pts[order]
        disp = disp[order]
    magnitude = np.linalg.norm(disp, axis=1)
    values = [magnitude, disp[:, 0], disp[:, 1], disp[:, 2]]
    titles = ["Total displacement", "U displacement", "V displacement", "W displacement"]
    labels = ["|U|", "U", "V", "W"]
    fig = plt.figure(figsize=(14, 11), dpi=int(cfg["dpi"]))
    for idx, (value, title, label) in enumerate(zip(values, titles, labels), start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        scatter = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=value,
            s=1.4,
            cmap="viridis" if idx == 1 else "coolwarm",
            linewidths=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("World X")
        ax.set_ylabel("World Y")
        ax.set_zlabel("World Z")
        _style_3d_axis(ax, pts, cfg)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_pair_surface_cloud_maps(
    output_dir: Path,
    frame_stem: str,
    pair_surfaces: list[dict[str, np.ndarray]],
    cfg: dict[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    patches = _collect_pair_surface_patches(pair_surfaces)
    if not patches:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    all_points = np.concatenate([points for points, _, _ in patches], axis=0)
    mins = np.nanmin(all_points, axis=0)
    maxs = np.nanmax(all_points, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.nanmax(maxs - mins)) * 0.55, 1.0e-9)

    fields = {
        "surface_cloud_morphology": {
            "title": "3D morphology cloud map",
            "label": "Z",
            "filename": f"{frame_stem}_surface_cloud_morphology.png",
            "cmap": "viridis",
            "signed": False,
            "getter": lambda points, displacement: points[:, 2],
        },
        "surface_cloud_displacement_total": {
            "title": "Total displacement cloud map",
            "label": "|U|",
            "filename": f"{frame_stem}_surface_cloud_displacement_total.png",
            "cmap": "plasma",
            "signed": False,
            "getter": lambda points, displacement: np.linalg.norm(displacement, axis=1),
        },
        "surface_cloud_displacement_ux": {
            "title": "Ux displacement cloud map",
            "label": "Ux",
            "filename": f"{frame_stem}_surface_cloud_displacement_ux.png",
            "cmap": "coolwarm",
            "signed": True,
            "getter": lambda points, displacement: displacement[:, 0],
        },
        "surface_cloud_displacement_uy": {
            "title": "Uy displacement cloud map",
            "label": "Uy",
            "filename": f"{frame_stem}_surface_cloud_displacement_uy.png",
            "cmap": "coolwarm",
            "signed": True,
            "getter": lambda points, displacement: displacement[:, 1],
        },
        "surface_cloud_displacement_uz": {
            "title": "Uz displacement cloud map",
            "label": "Uz",
            "filename": f"{frame_stem}_surface_cloud_displacement_uz.png",
            "cmap": "coolwarm",
            "signed": True,
            "getter": lambda points, displacement: displacement[:, 2],
        },
    }

    outputs: dict[str, str] = {}
    for key, spec in fields.items():
        values = [
            spec["getter"](points, displacement)[faces].mean(axis=1)
            for points, displacement, faces in patches
        ]
        merged = np.concatenate(values, axis=0)
        norm = Normalize(*_surface_cloud_color_limits(merged, signed=bool(spec["signed"])))
        cmap = plt.get_cmap(str(spec["cmap"]))

        fig = plt.figure(figsize=(9.2, 7.4), dpi=int(cfg["dpi"]))
        ax = fig.add_subplot(111, projection="3d")
        for points, displacement, faces in patches:
            point_values = spec["getter"](points, displacement)
            face_values = point_values[faces].mean(axis=1)
            collection = Poly3DCollection(
                points[faces],
                facecolors=cmap(norm(face_values)),
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
            )
            ax.add_collection3d(collection)

        ax.set_title(str(spec["title"]))
        ax.set_xlabel("World X")
        ax.set_ylabel("World Y")
        ax.set_zlabel("World Z")
        ax.view_init(elev=float(cfg["view_elev"]), azim=float(cfg["view_azim"]))
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.grid(True, color="#d0d7de", linewidth=0.5, alpha=0.8)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
        cbar.set_label(str(spec["label"]))
        fig.tight_layout()
        path = output_dir / str(spec["filename"])
        fig.savefig(path)
        plt.close(fig)
        outputs[key] = str(path)
    return outputs


def _collect_pair_surface_patches(
    pair_surfaces: list[dict[str, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for surface in pair_surfaces:
        points = np.asarray(surface["points_ref_world"], dtype=np.float64)
        displacement = np.asarray(surface["displacement_world"], dtype=np.float64)
        faces = np.asarray(surface["faces"], dtype=np.int32)
        valid_faces = np.asarray(surface["valid_faces"], dtype=bool)
        valid_points = np.asarray(surface.get("valid_points", np.ones(len(points), dtype=bool)), dtype=bool)
        if "outlier_filter_keep" in surface:
            valid_points &= np.asarray(surface["outlier_filter_keep"], dtype=bool)
        finite_points = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(displacement), axis=1)
        valid_points &= finite_points
        keep_faces = valid_faces & np.all(valid_points[faces], axis=1)
        if np.any(keep_faces):
            patches.append((points, displacement, faces[keep_faces]))
    return patches


def _surface_cloud_color_limits(values: np.ndarray, signed: bool) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        bound = float(np.nanpercentile(np.abs(values), 98))
        bound = max(bound, 1.0e-12)
        return -bound, bound
    vmin, vmax = np.nanpercentile(values, [2, 98])
    if vmax - vmin <= 1.0e-12:
        center = 0.5 * float(vmin + vmax)
        return center - 0.1, center + 0.1
    return float(vmin), float(vmax)


def _count_valid_pair_surface_faces(pair_surfaces: list[dict[str, np.ndarray]]) -> int:
    return sum(faces.shape[0] for _, _, faces in _collect_pair_surface_patches(pair_surfaces))


def _interpolate_cylinder_surface_fields(
    points: np.ndarray,
    displacement: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    from scipy.interpolate import griddata

    theta = np.arctan2(points[:, 2], points[:, 0])
    y = points[:, 1]
    radius = np.linalg.norm(points[:, [0, 2]], axis=1)
    magnitude = np.linalg.norm(displacement, axis=1)
    theta_grid = np.linspace(-np.pi, np.pi, int(cfg["surface_theta_samples"]))
    y_grid = np.linspace(float(np.min(y)), float(np.max(y)), int(cfg["surface_y_samples"]))
    theta_mesh, y_mesh = np.meshgrid(theta_grid, y_grid)

    theta_aug = np.concatenate([theta - 2.0 * np.pi, theta, theta + 2.0 * np.pi])
    y_aug = np.tile(y, 3)
    coords = np.column_stack([theta_aug, y_aug])

    def interp(values: np.ndarray) -> np.ndarray:
        values_aug = np.tile(values, 3)
        linear = griddata(coords, values_aug, (theta_mesh, y_mesh), method="linear")
        if np.any(~np.isfinite(linear)):
            nearest = griddata(coords, values_aug, (theta_mesh, y_mesh), method="nearest")
            linear = np.where(np.isfinite(linear), linear, nearest)
        return linear

    radius_grid = interp(radius)
    return {
        "theta": theta_mesh,
        "x": radius_grid * np.cos(theta_mesh),
        "y": y_mesh,
        "z": radius_grid * np.sin(theta_mesh),
        "radius": radius_grid,
        "magnitude": interp(magnitude),
        "ux": interp(displacement[:, 0]),
        "uy": interp(displacement[:, 1]),
        "uz": interp(displacement[:, 2]),
    }


def _style_3d_axis(ax: Any, points: np.ndarray, cfg: dict[str, Any]) -> None:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=float(cfg["view_elev"]), azim=float(cfg["view_azim"]))
    if points.size:
        mins = np.nanmin(points, axis=0)
        maxs = np.nanmax(points, axis=0)
        center = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.55, 1.0e-9)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.grid(True, color="#d0d7de", linewidth=0.5, alpha=0.8)


def _style_surface_axis(ax: Any, surface: dict[str, np.ndarray], cfg: dict[str, Any]) -> None:
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")
    ax.zaxis.labelpad = -1.0
    ax.view_init(elev=float(cfg["view_elev"]), azim=float(cfg["view_azim"]))
    points = np.column_stack([surface["x"].ravel(), surface["y"].ravel(), surface["z"].ravel()])
    _style_3d_axis(ax, points, cfg)


def _field_color_limits(values: np.ndarray, signed: bool) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if signed:
        bound = max(abs(vmin), abs(vmax), 1.0e-12)
        return -bound, bound
    if vmax - vmin <= 1.0e-12:
        center = 0.5 * (vmin + vmax)
        return center - 0.1, center + 0.1
    return vmin, vmax


def _write_reference_points_ply(path: Path, points: np.ndarray, valid: np.ndarray) -> Path:
    pts = points[valid]
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(pts)}\n")
        handle.write("property double x\nproperty double y\nproperty double z\n")
        handle.write("end_header\n")
        for point in pts:
            handle.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")
    return path


def _write_surface_fields_npz(path: Path, surface: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **surface)
    return path


def _array_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }
