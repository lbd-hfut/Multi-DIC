# Native Recon3D

This directory contains the optional pybind11 acceleration module for the
Multi-DIC `recon3d` step.

The Python workflow in `multidic/recon3d.py` first tries to import:

```python
import native_recon3d
```

If the module is not available, the same track-anchored reconstruction is run
with the NumPy fallback.

## Build

The project environment is expected to provide `cmake`, `ninja`, and
`pybind11` as listed in `environment.yml`.

Example:

```bash
cmake -S . -B build/native -G Ninja
cmake --build build/native
```

WSL example using the project mamba environment:

```bash
export PATH="$HOME/.local/share/mamba/envs/multi-dic/bin:$PATH"
cmake -S . -B build/wsl-native -G Ninja \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE=$HOME/.local/share/mamba/envs/multi-dic/bin/python \
  -Dpybind11_DIR=$($HOME/.local/share/mamba/envs/multi-dic/bin/python -m pybind11 --cmakedir)
cmake --build build/wsl-native --target native_recon3d
PYTHONPATH=build/wsl-native/native/recon3d:. \
  python -m multidic run --config configs/MDIC.yaml --step recon3d
python native/recon3d/tests/compare_native_numpy.py \
  --repo /mnt/c/01project/Multi-DIC \
  --native-dir /mnt/c/01project/Multi-DIC/build/wsl-native/native/recon3d \
  --frame 002.bmp
```

The root `CMakeLists.txt` adds this directory with `add_subdirectory`. If
`pybind11` is missing, CMake skips `native_recon3d` instead of failing the whole
native build.

## Exposed Function

```python
native_recon3d.reconstruct_tracks(
    K, dist, R, t,
    point_indices, cam_indices, uv,
    dic_u, dic_v, dic_corrcoef, dic_valid,
    reduced_height, reduced_width, subset_spacing,
    min_views, min_corrcoef, max_reprojection_error_px,
    scale,
)
```

It returns the same dictionary keys as the NumPy fallback:

- `point_indices`
- `points_ref_sfm`, `points_def_sfm`, `displacement_sfm`
- `points_ref_world`, `points_def_world`, `displacement_world`
- `num_views`, `mean_corrcoef`
- `reprojection_error_ref`, `reprojection_error_def`
- `valid`
