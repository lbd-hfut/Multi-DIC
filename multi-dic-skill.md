# Multi-DIC Agent Runbook

This note is for coding agents working in this repository. It explains when a
local native build is needed, how to set up the build environment on the current
platform, how to compile the native Python API extensions from `native/`, and
how to run and debug real experiment data when no ground truth is available.

## Repository Contract

- The supported native build entry point is `native/CMakeLists.txt`.
- Do not create separate experimental build branches or alternate native build
  layouts unless the user explicitly asks for that.
- The Python package surface is `pymultidic`.
- PyMultiDIC 2.x publishes CPython 3.12 wheels for Windows x86_64 and Linux
  x86_64 (`manylinux_2_39`, glibc 2.39 or newer). Other Python versions, older
  Linux distributions, and macOS require an explicitly supported
  source-development setup.
- The default SfM backend is `native_colmap`, a CPU-only embedded COLMAP
  adapter. It is the project's only SfM implementation.
- If the user is using an installed PyPI wheel (`pip install pymultidic`) and
  is not changing native source code, do not set up a local C++ build
  environment. Validate the installed package and run the Python API directly.
- The native outputs expected after a source build are:
  - `build/wsl-native/ncorr/ncorr_cli`
  - `build/wsl-native/recon3d/native_recon3d*.so`
  - `build/wsl-native/colmap/native_colmap*.so`
  - or the equivalent `build/windows-native/...` `.exe` / `.pyd` files on
    Windows.
  - on this Windows checkout, the validated COLMAP source-port build also
    produces `build/native-colmap-port/colmap/native_colmap*.pyd` and
    `build/native-colmap-port/colmap/mdic_colmap_sparse.lib`.

## Choose Install Mode First

Before installing compilers or running CMake, decide whether this is a package
use case or a source-development use case.

Use the PyPI/package path when:

- The user only wants to run Multi-DIC on experiment data.
- The user installed with `pip install pymultidic`.
- The user is not editing files under `native/`.
- The installed wheel already imports the native modules successfully.

In that case, do not prepare a local native build environment. Use:

```bash
python -m pip install -U pymultidic
python -c "import pymultidic; print('pymultidic import ok')"
```

Then run the normal Python workflow or CLI against the experiment config.

Use the source-build path only when:

- The user cloned the repository and wants to run from checkout.
- The user changed C++ code under `native/`.
- The platform does not have a usable wheel.
- You are validating wheel/build behavior before release.
- `native_colmap`, `native_recon3d`, or `ncorr_cli` is missing from the current
  source checkout.

## Choose The Platform Path

First identify the platform you are actually running commands in:

```bash
python -c "import platform, sys; print(platform.system(), platform.machine(), sys.executable)"
```

Use WSL/Linux when available for native development because the COLMAP CPU
dependencies are easier to install there. Use Windows only when validating a
Windows wheel or a Windows local build.

## WSL / Linux Environment

From the repository root:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build python3-dev python3-pip python3-venv \
  libboost-graph-dev libeigen3-dev libceres-dev \
  libsqlite3-dev libgoogle-glog-dev libmetis-dev libsuitesparse-dev

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pybind11 scikit-build-core
```

Build from `native/` only:

```bash
cmake -S native -B build/wsl-native -G Ninja \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build/wsl-native -j2
```

If you are calling from Windows PowerShell into WSL, use:

```powershell
wsl -e bash -lc "cd /mnt/c/01project/Multi-DIC && cmake -S native -B build/wsl-native -GNinja -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=/usr/bin/python3 -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && cmake --build build/wsl-native -j2"
```

If command substitution inside PowerShell is awkward, resolve the pybind11 CMake
path first in WSL:

```bash
python3 -m pybind11 --cmakedir
```

Then pass the printed path as `-Dpybind11_DIR=...`.

## Windows Environment

Use a Developer PowerShell or a shell where a C++ compiler, CMake, and Ninja are
available.

```powershell
python -m pip install -U pybind11 scikit-build-core cmake ninja
cmake -S native -B build/windows-native -G Ninja -DPYBIND11_FINDPYTHON=ON
cmake --build build/windows-native
```

For the current local Windows COLMAP source-port validation tree, use the
project-local Visual Studio Build Tools and explicit pybind11 path:

```powershell
cmd /c ""C:\01project\ncorr\.tools\vsbt\VC\Auxiliary\Build\vcvars64.bat" >nul && python -m cmake -S native -B build\native-colmap-port -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\LBD\AppData\Roaming\Python\Python313\Scripts\ninja.exe -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=OFF -Dpybind11_DIR=C:\Users\LBD\AppData\Roaming\Python\Python313\site-packages\pybind11\share\cmake\pybind11"
cmd /c ""C:\01project\ncorr\.tools\vsbt\VC\Auxiliary\Build\vcvars64.bat" >nul && C:\Users\LBD\AppData\Roaming\Python\Python313\Scripts\ninja.exe -C build\native-colmap-port native_colmap -j 2"
```

If CMake cannot find a compiler, install Visual Studio Build Tools or use a
Developer PowerShell. If dependency discovery fails on Windows, prefer WSL for
development and keep Windows for wheel validation.

## Conda Environment

When using conda or mamba:

```bash
mamba env create -f environment.yml
mamba activate multi-dic
```

Then run the platform build command above. `environment.yml` includes the
native CPU build dependencies from conda-forge.

## Real Experiment Case Layout

Real experiment cases do not need ground truth. A valid case can contain only
speckle images and checkerboard calibration images:

```text
case/MyExperiment/
  images/
    cam_0/
      001.bmp        # reference speckle frame
      002.bmp        # deformed speckle frame
      ...
    cam_1/
      001.bmp
      002.bmp
      ...
  calibrate_images/
    chessboard_meta.json
    cam_0/
      001.bmp
      ...
    cam_1/
      001.bmp
      ...
```

`ground_truth/` is not required for real experiments and should not be used for
quality checks unless the user explicitly provides simulated truth data.

Camera folder names should match the configured `camera_glob`, usually
`cam_*`. The same reference and deformed frame names should exist under every
camera folder.

## Minimal Real-Data Config

Copy `configs/MDIC.yaml` and edit at least these fields:

```yaml
project:
  name: MyExperiment
  case_root: case/MyExperiment
  output_root: results

data:
  speckle_dir: images
  calibration_dir: calibrate_images
  camera_glob: cam_*
  reference_frame: "001.bmp"
  deformed_frames:
    - "002.bmp"

scale_correction:
  checkerboard_meta: calibrate_images/chessboard_meta.json
  image_dir: calibrate_images
  square_size: 10.0
  square_size_unit: mm

colmap:
  matcher: exhaustive
  matching_window: 2
  wrap_matching: true
  max_features: 20000
  random_seed: 1
  spatial_outlier_filter:
    enabled: true
    radius_mad_z: 8.0
    knn: 8
    knn_mad_z: 8.0

dic2d:
  subset_radius: 20
  subset_spacing: 5
  format:
    cutoff_corrcoef: 0.6

recon3d:
  min_views: 2
  min_corrcoef: 0.6
  max_reprojection_error_px: 2.0
```

The native selector requires every configured camera and reports per-camera
observation coverage.

## Run The Pipeline

Run everything:

```bash
python run.py --config configs/MDIC.yaml
```

Run one step at a time from Python when debugging:

```python
import pymultidic

config = pymultidic.load_config("configs/MDIC.yaml")
pymultidic.run_validate(config)
pymultidic.run_sfm(config)
pymultidic.run_scale(config)
pymultidic.run_mask(config)
pymultidic.run_dic2d(config)
pymultidic.run_recon3d(config)
pymultidic.run_visualize3d(config)
```

Pipeline order:

```text
validate -> sfm -> scale -> mask -> dic2d -> recon3d -> visualize3d
```

Do not skip `scale` for real data if physical units matter. Without scale, SfM
coordinates are arbitrary up to a scale factor.

## What To Inspect Without Ground Truth

For real experiments, judge quality using internal consistency and visual
diagnostics:

### 1. Validation

Report:

```text
case/<name>/results/logs/validation_report.json
```

Check:

- All expected cameras are found.
- Reference and deformed frames exist for every camera.
- Image sizes are consistent.
- No missing checkerboard metadata if scale correction is enabled.

### 2. SfM

Reports and figures:

```text
results/logs/sfm_report.json
results/sfm/colmap/sparse_scene.png
results/sfm/colmap/camera_observations_2d.png
results/sfm/colmap/camera_observations_3d.png
results/sfm/colmap/sparse_point_filter.json
```

Good signs:

- `backend` is `native_colmap`.
- `native_colmap_backend` is `embedded_colmap_sparse_source`.
- `native_colmap_capabilities.implementation` is
  `colmap_sparse_source_port`.
- One selected model contains every configured camera.
- `missing_cameras` is empty.
- Per-camera 2D observation counts are not zero or single-digit.
- `sparse_point_filter.json` reports a conservative number of removed 3D
  sparse outliers and `fallback_used: false`.
- Mean reprojection error is low enough for the experiment resolution, commonly
  around sub-pixel to a few pixels.

If SfM is bad:

- Confirm camera folder order follows the physical camera ring/order.
- Keep `matcher: exhaustive` for the 12-camera cylinder case unless you are
  intentionally trading accuracy for speed.
- Inspect native candidate diagnostics when a camera fails registration or has
  weak observations.
- Increase `max_features` if texture is rich but matches are sparse.
- Check `camera_observations_2d.png`; every camera should have broad red-point
  coverage over the speckle region.
- Check `sparse_scene.png`; camera centers should form one continuous rig, not
  a split or flying layout.

### 3. Scale Correction

Report:

```text
results/logs/scale_report.json
```

Good signs:

- Enough cameras detect the checkerboard.
- Enough common corners are triangulated.
- Reprojection error is low.
- Edge coefficient of variation is small.
- `sfm_to_world_scale` is positive and stable.

If scale is bad:

- Verify `square_size` and units.
- Check `calibrate_images/chessboard_meta.json`.
- Inspect corner overlay images under `results/scale/detections`.
- If checkerboard views are partially occluded, raise `min_common_corners`
  carefully or remove unusable calibration images.

### 4. Masks

Reports and figures:

```text
results/logs/mask_report.json
results/masks/auto_roi_summary.png
results/masks/overlay/
results/masks/debug/
```

Good signs:

- The ROI covers the speckled specimen and excludes most background.
- No important specimen region is cut away.
- Overlays align with each camera image.

If masks are bad:

- Provide user masks under `case/<name>/masks` and set
  `mask.use_user_mask_if_present: true`.
- Adjust `external_threshold`, `component_radius_scale`, `edge_scale`, and
  `min_hole_area` only after viewing overlays.
- Do not tune masks from final displacement plots alone; inspect per-camera ROI
  overlays first.

### 5. 2D DIC

Report:

```text
results/logs/dic2d_report.json
results/dic2d/dic2d_<camera>_<frame>.npz
```

Good signs:

- Each camera produces a DIC output for each deformed frame.
- Correlation coefficients are mostly above `cutoff_corrcoef`.
- The valid region follows the mask rather than random islands.

If DIC is unstable:

- Increase `subset_radius` for low-texture or noisy images.
- Increase `subset_spacing` for a faster, coarser first pass.
- Increase `seed_search_radius` when displacement between frames is large.
- Lower `cutoff_corrcoef` only when image quality justifies it; otherwise it
  can keep bad matches.
- Check image exposure and blur. Algorithm tuning cannot fix severe motion blur
  or saturation.

### 6. 3D Reconstruction

Reports and products:

```text
results/logs/recon3d_report.json
results/recon3d/recon3d_<frame>.npz
results/recon3d/recon3d_<frame>.ply
results/recon3d/qc/<frame>/
```

Good signs:

- `num_tracks_valid / num_tracks_total` is high.
- Mean number of views is at least near 2 and preferably higher.
- Reference and deformed reprojection errors are low.
- Camera contributions are not dominated by one camera.
- Outlier filter removes only a small fraction unless the experiment is noisy.

If 3D reconstruction is bad:

- Relax `max_reprojection_error_px` gradually.
- Lower `min_corrcoef` only if DIC quality is visually acceptable.
- Check SfM camera geometry before changing recon3d thresholds.
- Check whether scale correction produced an unreasonable scale.
- Inspect per-camera contributions to identify failing cameras.

### 7. Visualization

Figures:

```text
results/figures/*initial_shape*.png
results/figures/*surface_fields*.png
results/figures/surface_clouds/*morphology*.png
results/figures/surface_clouds/*displacement*.png
```

Good signs:

- The morphology resembles the specimen.
- Displacement fields are spatially coherent.
- Component maps have plausible directionality for the loading case.
- No isolated high-magnitude islands dominate the color scale.

Without ground truth, do not claim accuracy from plots alone. State that the
result passes internal consistency checks and list the residual risks.

## Recommended Debug Order

When a real experiment fails, debug upstream first:

```text
folder layout -> validate -> SfM -> scale -> masks -> DIC2D -> recon3D -> visualization
```

Avoid tuning later thresholds to compensate for earlier failures. A bad SfM
model will poison scale and 3D reconstruction. A bad mask will poison DIC2D.
Bad DIC2D will poison recon3D.

## Common Failure Patterns

### Native extension import fails

Symptoms:

- `ModuleNotFoundError: native_colmap`
- `ModuleNotFoundError: native_recon3d`
- DIC cannot find `ncorr_cli`

Actions:

If this is a PyPI wheel installation, first verify the installed package:

```bash
python -m pip show pymultidic
python -c "import pymultidic; print('pymultidic import ok')"
```

Do not ask the user to compile locally unless the installed wheel is missing the
required native module for their platform or the user is intentionally working
from source.

If this is a source checkout, rebuild from `native/`:

```bash
cmake --build build/wsl-native -j2
python -c "import sys; sys.path[:0]=['build/wsl-native/colmap','build/wsl-native/recon3d']; import native_colmap, native_recon3d; print(native_colmap.capabilities())"
```

For Windows, replace `build/wsl-native` with `build/windows-native`, or use the
validated `build/native-colmap-port` COLMAP source-port command shown above.
Expected native COLMAP capabilities include:

```text
implementation = colmap_sparse_source_port
mapping = colmap_correspondence_graph_incremental_mapper
refinement = colmap_local_and_global_ceres_bundle_adjustment
```

### SfM creates multiple partial models

Actions:

- Require every configured camera in the selected native model.
- Confirm camera naming order is physical order.
- Inspect the multi-initial-pair candidate diagnostics in
  `results/logs/sfm_report.json`.
- Check `camera_observations_2d.png`, `camera_observations_3d.png`, and
  `sparse_scene.png`.
- Keep the selected model only if all configured cameras are registered, each
  camera has reasonable observation coverage, and camera centers have no
  obvious outliers.

### Checkerboard scale fails

Actions:

- Verify square size and units.
- Confirm calibration images are under each camera folder.
- Inspect `results/scale/detections/*_corners.png`.
- Make sure the checkerboard is visible in enough cameras with common corners.

### DIC results are sparse or noisy

Actions:

- Inspect mask overlays first.
- Increase `subset_radius`.
- Increase `seed_search_radius` for large deformation.
- Use coarser `subset_spacing` for a quick diagnostic run.
- Keep `cutoff_corrcoef` conservative unless visual checks support lowering it.

### 3D displacement has isolated spikes

Actions:

- Keep outlier filtering enabled.
- Inspect per-camera DIC quality.
- Tighten `max_reprojection_error_px` if spikes are geometric mismatches.
- Raise `min_corrcoef` if spikes are low-correlation DIC matches.
- Confirm rigid body motion removal is appropriate for the experiment.

## Reporting Results To The User

For real experiments without truth data, report:

- Pipeline status and completed steps.
- Number of cameras found and registered.
- SfM model count, registered cameras, sparse point count, and reprojection
  error.
- Scale quality: detected cameras, triangulated corners, reprojection error,
  and scale value.
- DIC validity and correlation quality.
- Recon3D valid track ratio, view count statistics, reprojection error, and
  displacement summary.
- Paths to the key figures and reports.
- Any residual uncertainty, especially poor overlap, weak checkerboard
  detection, low correlation, or missing cameras.

Avoid saying the result is "accurate" when there is no ground truth. Prefer:

```text
The result is internally consistent under the current SfM, scale, DIC, and
reprojection checks.
```

## Cleanup Rules

Do not commit local build caches or raw generated case results unless the user
explicitly asks for them. Keep these ignored:

- `build/`
- `native/**/build*/`
- `_skbuild/`
- `__pycache__/`
- `case/**/results/`
- compiler products such as `*.o`, `*.a`, `*.so`, `*.pyd`, `*.dll`

It is acceptable to copy selected, lightweight demonstration outputs into
`docs/results/...` when updating the README or manual.
