from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare native_recon3d against the NumPy fallback.")
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument("--native-dir", required=True, help="Directory containing native_recon3d .so/.pyd.")
    parser.add_argument("--config", default="configs/MDIC.yaml", help="Config path relative to repo.")
    parser.add_argument("--frame", default="002.bmp", help="Deformed frame to compare.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(Path(args.native_dir).resolve()))
    sys.path.insert(0, str(repo))

    import native_recon3d
    from multidic.config import load_config
    from multidic.recon3d import (
        _load_dic2d_fields,
        _load_npz,
        _load_sfm_to_world_scale,
        _recon_config,
        _reconstruct_tracks_numpy,
    )

    config = load_config(repo / args.config, workspace_root=repo)
    cfg = _recon_config(config)
    cameras = _load_npz(cfg["sfm_dir"] / "cameras.npz")
    observations = _load_npz(cfg["sfm_dir"] / "observations.npz")
    cam_names = [str(name) for name in cameras["cam_names"]]
    dic = _load_dic2d_fields(cfg["dic2d_dir"], cam_names, args.frame)
    scale = _load_sfm_to_world_scale(config, cfg)

    numpy_result = _reconstruct_tracks_numpy(cameras, observations, dic, cfg, scale)
    native_result = native_recon3d.reconstruct_tracks(
        np.asarray(cameras["K"], dtype=np.float64),
        np.asarray(cameras["dist"], dtype=np.float64),
        np.asarray(cameras["R"], dtype=np.float64),
        np.asarray(cameras["t"], dtype=np.float64),
        np.asarray(observations["point_indices"], dtype=np.int64),
        np.asarray(observations["cam_indices"], dtype=np.int32),
        np.asarray(observations["uv"], dtype=np.float64),
        dic["u"],
        dic["v"],
        dic["corrcoef"],
        dic["valid"],
        int(dic["reduced_height"]),
        int(dic["reduced_width"]),
        int(dic["subset_spacing"]),
        int(cfg["min_views"]),
        float(cfg["min_corrcoef"]),
        float(cfg["max_reprojection_error_px"]),
        float(scale),
    )

    for name, result in (("numpy", numpy_result), ("native", native_result)):
        valid = np.asarray(result["valid"], dtype=bool)
        disp_norm = np.linalg.norm(np.asarray(result["displacement_world"])[valid], axis=1)
        print(
            f"{name}: valid={int(np.count_nonzero(valid))}/{len(valid)} "
            f"median_norm={float(np.median(disp_norm)):.12g}"
        )

    common = np.asarray(numpy_result["valid"], dtype=bool) & np.asarray(native_result["valid"], dtype=bool)
    diff = np.linalg.norm(
        np.asarray(numpy_result["displacement_world"])[common] - np.asarray(native_result["displacement_world"])[common],
        axis=1,
    )
    print(f"common_valid={int(np.count_nonzero(common))}")
    print(f"median_abs_disp_diff={float(np.median(diff)):.12g}")
    print(f"p95_abs_disp_diff={float(np.percentile(diff, 95)):.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
