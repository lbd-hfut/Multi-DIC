# Multi-DIC native COLMAP sparse SfM

This directory contains the only SfM implementation supported by Multi-DIC.
It builds an embedded CPU-only `native_colmap` extension and never invokes an
external executable or another Python SfM backend.

The runtime pipeline is:

```text
reference images -> CPU SIFT and verified matches
                 -> COLMAP CorrespondenceGraph and IncrementalMapper
                 -> IncrementalTriangulator
                 -> COLMAP local/global Ceres bundle adjustment
                 -> COLMAP text model and Multi-DIC products
```

`src/colmap/` is the trimmed COLMAP sparse source set actually linked into
`mdic_colmap_sparse`. `src/thirdparty/PoseLib/` contains only the files needed
by that build. Dense MVS, GPU, GUI, meshing, retrieval, command-line programs,
and unrelated source trees are intentionally absent.

The normal project build is sufficient:

```powershell
python -m cmake -S native -B build/native-colmap-port -G Ninja
cmake --build build/native-colmap-port --target native_colmap -j 2
```

The build fails during configuration when the project-local native COLMAP
dependencies are incomplete; there is no alternate SfM implementation.

The stable Python entry point is `native_colmap.run_cpu_sfm(...)`. It writes
COLMAP text models plus the camera, sparse-point, observation, PLY, MAT, NPZ,
JSON, and visualization products consumed by Multi-DIC.

COLMAP's BSD license is reproduced in `LICENSE.COLMAP.txt`.
