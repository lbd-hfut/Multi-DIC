# Native Ncorr Port

This directory is the runtime Ncorr implementation for Multi-DIC.

Reference source:

```text
reference_code_lib/ncorr_2D_matlab/
```

Porting contract:

- Preserve the original Ncorr C++ algorithm modules as much as possible.
- Remove MATLAB GUI dependencies.
- Expose a function-style DIC interface that receives:
  - reference grayscale image
  - deformed grayscale image
  - ROI mask
  - seed point selected from COLMAP observations inside the ROI
  - `dic2d` config parameters
- Do not include Ncorr strain calculation in this module.

Initial C++ files to port:

- `standard_datatypes.cpp/.h`
- `ncorr_datatypes.cpp/.h`
- `ncorr_lib.cpp/.h`
- `ncorr_alg_calcseeds.cpp`
- `ncorr_alg_rgdic.cpp`

MATLAB GUI files are design references only. Their parameter choices should be
represented in `configs/MDIC.yaml` under `dic2d`.

## WSL build

When CMake is unavailable in WSL, the current API scaffold can be checked with:

```bash
cd /mnt/c/01project/Multi-DIC/native/ncorr
make smoke
make all
```

This builds `build/libnative_ncorr.a` and runs a small seed-selection/API smoke
test. `make all` also builds `build/ncorr_cli`, a thin command-line wrapper used
by the Python `dic2d` step to call the function-style native API from WSL. The
legacy Ncorr MEX files in `legacy/` are preserved source material and are not
compiled by this Makefile yet.

## Current status

Implemented:

- legacy core files copied into `legacy/` without modification
- no-GUI public API types in `include/ncorr_api.h`
- ROI-contained seed selection helper
- COLMAP observation to Ncorr reduced-grid seed snapping
- reduced reference-grid point generation
- retained ROI region extraction with Ncorr-style 4-connected components and minimum-area filtering
- single-thread initial `SeedInfo` and reduced-grid thread diagram preparation
- seed-level NCC initial displacement guess within `dic2d.seed_search_radius`
- reduced-grid queue propagation following the original RG-DIC neighbor-expansion flow
- six-parameter affine IC-GN refinement after NCC initialization for each reduced-grid point
- compressed `.npz` displacement output from the Python workflow (`x_ref`, `y_ref`, `x_def`, `y_def`, `u`, `v`, `ux`, `uy`, `vx`, `vy`, `corrcoef`, `valid`)
- input validation for the future `run_ncorr_dic2d` call
- native `ncorr_cli` wrapper for reference/deformed image, ROI, seed and config inputs
- Python `dic2d` step that resolves ROI masks, selects one COLMAP observation per camera, snaps it to the Ncorr grid, calls the WSL native wrapper when available, and reports per-frame seed initial guesses

Pending:

- extract the computation body from `legacy/ncorr_alg_calcseeds.cpp`
- extract the computation body from `legacy/ncorr_alg_rgdic.cpp`
- replace MEX `mxArray` inputs/outputs with native structs
- replace the current bilinear interpolation path with original-style biquintic interpolation
