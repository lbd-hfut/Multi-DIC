# Multi-DIC

Multi-DIC is a config-driven, calibration-free multi-view DIC workflow. The
current SfM/self-calibration stage uses the official `pycolmap` Python package
with a pinned version for reproducibility.

## Quick Start

```bash
~/.local/bin/micromamba create -y -f environment.yml
~/.local/bin/micromamba run -n multi-dic python -m pip install -e .
~/.local/bin/micromamba run -n multi-dic python -m pymultidic run --config configs/MDIC.yaml --step validate
~/.local/bin/micromamba run -n multi-dic python -m pymultidic run --config configs/MDIC.yaml --step sfm
~/.local/bin/micromamba run -n multi-dic python -m pymultidic run --config configs/MDIC.yaml --step scale
~/.local/bin/micromamba run -n multi-dic python -m pymultidic run --config configs/MDIC.yaml --step mask
~/.local/bin/micromamba run -n multi-dic python -m pymultidic run --config configs/MDIC.yaml --step dic2d
~/.local/bin/micromamba run -n multi-dic python -m pymultidic run --config configs/MDIC.yaml --step recon3d
```

`reference_code_lib/` is only used as local reference source code. The formal
project implementation lives in the repository root.

The same flow can be called from Python:

```python
import pymultidic

config = pymultidic.load_config("configs/MDIC.yaml")
pymultidic.run_pipeline(config, steps=["validate", "sfm", "scale", "mask", "dic2d", "recon3d"])
```

## Native Build

The native C++ components can be built from WSL through the top-level
`native/CMakeLists.txt`. One configure/build pass produces the Ncorr library and
CLI plus the Recon3D pybind11 extension:

```bash
cmake -S native -B build/wsl-native -G Ninja \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE=/usr/bin/python3 \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build/wsl-native
```

Expected outputs include:

- `build/wsl-native/ncorr/libnative_ncorr.a`
- `build/wsl-native/ncorr/ncorr_cli`
- `build/wsl-native/recon3d/native_recon3d*.so`

## COLMAP Backend

The COLMAP stage is selected in `configs/MDIC.yaml`:

```yaml
colmap:
  backend: pycolmap
  workspace: colmap
  camera_model: SIMPLE_RADIAL
  matcher: exhaustive
  use_gpu: false
  overwrite: true
```

The project pins `pycolmap==4.1.0` in both `environment.yml` and
`pyproject.toml` to reduce interface drift. SfM results are written under the
case result directory. Following the NDeF-DIC style, each camera folder's first
speckle image is copied into a flat `colmap_images/` directory before running
COLMAP. Example outputs:

- `case/CylinderDIC/results/logs/sfm_report.json`
- `case/CylinderDIC/results/sfm/colmap/colmap.db`
- `case/CylinderDIC/results/sfm/colmap/colmap_images/`
- `case/CylinderDIC/results/sfm/colmap/colmap_sfm/`
- `case/CylinderDIC/results/sfm/colmap/cameras.npz`
- `case/CylinderDIC/results/sfm/colmap/cameras.mat`
- `case/CylinderDIC/results/sfm/colmap/sparse_points.npz`
- `case/CylinderDIC/results/sfm/colmap/points3D.mat`
- `case/CylinderDIC/results/sfm/colmap/observations.npz`
- `case/CylinderDIC/results/sfm/colmap/sparse_scene.png`

## Scale Correction

The `scale` step follows the NDeF-DIC `sfm2world` chessboard workflow. It reads
`cameras.npz`, detects chessboard inner corners in `calibrate_images/cam_*`,
triangulates the board corners, and estimates the physical scale:

- `case/CylinderDIC/results/scale/sfm2world_scale.json`
- `case/CylinderDIC/results/scale/chessboard_triangulation.npz`
- `case/CylinderDIC/results/scale/detections/`

## ROI Masks

The `mask` step follows the NDeF-DIC automatic ROI workflow. If
`mask.user_mask_dir` contains one mask for every registered camera, those masks
are used directly. Otherwise, Multi-DIC builds masks from SfM observations and
reference-image speckle texture:

- `case/CylinderDIC/results/logs/mask_report.json`
- `case/CylinderDIC/results/masks/mask/`
- `case/CylinderDIC/results/masks/overlay/`
- `case/CylinderDIC/results/masks/debug/`
- `case/CylinderDIC/results/masks/auto_roi_meta.json`
- `case/CylinderDIC/results/masks/auto_roi_summary.png`

## 3D Reconstruction

The `recon3d` step uses SfM track ids as cross-camera anchors. It samples each
camera's DIC2D displacement field at the COLMAP observation, triangulates the
reference and deformed positions, and exports sparse 3D displacement points:

- `case/CylinderDIC/results/logs/recon3d_report.json`
- `case/CylinderDIC/results/recon3d/recon3d_002.npz`
- `case/CylinderDIC/results/recon3d/recon3d_002.ply`
- `case/CylinderDIC/results/recon3d/qc/002/*_hist.png`
- `case/CylinderDIC/results/recon3d/qc/002/*_colored.ply`
- `case/CylinderDIC/results/recon3d/qc/002/*_vectors.ply`

If the optional `native_recon3d` pybind11 module is available, it is used for
the expensive interpolation and triangulation loop. Otherwise the workflow uses
the same NumPy implementation.

Recon3D also writes QC statistics and figures for displacement norm,
reprojection error, DIC correlation, view count, and per-camera contribution.
After triangulation, Recon3D applies a configurable 3D outlier filter to remove
large isolated reconstructed points based on robust position-radius and
displacement-norm MAD thresholds. The filter updates the valid masks and records
removed counts in `recon3d_report.json`.

Recon3D additionally exports MultiDIC-style pair surfaces under
`case/CylinderDIC/results/recon3d/pairs/<frame>/`. By default
`recon3d.pairs.mode: auto_spatial` orders cameras from `camera_centers_world`.
Circular layouts are connected with wrap; non-circular layouts only connect
spatial neighbors without forcing the two ends into a pair. This can be
overridden in `configs/MDIC.yaml` with `recon3d.pairs.mode: manual`.

MultiDIC-style post-processing for each pair is written under
`case/CylinderDIC/results/recon3d/post/<frame>/`. It includes raw 3D
displacement, rigid-body-motion-removed displacement (`ARBM`), rigid transform
parameters, face centroids, face correlation, and face isotropy.
