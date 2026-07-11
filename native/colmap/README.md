# Multi-DIC native COLMAP scope

This module intentionally exposes only the CPU sparse SfM surface that Multi-DIC
needs:

- SIFT feature extraction for the selected reference images.
- Exhaustive pair matching and two-view geometry verification.
- Incremental sparse mapping with bundle adjustment.
- Binary and text COLMAP model export for cameras, sparse points, and tracks.

It intentionally excludes GPU/CUDA SIFT, GUI, dense/MVS reconstruction, meshing,
vocabulary-tree retrieval, and COLMAP's Python API. Multi-DIC consumes the
resulting COLMAP text model through its own stable exporter in
`multidic.colmap_backends.native_colmap_backend`.

Build modes:

- Embedded mode links the pybind11 module against `native/colmap/upstream` by
  default. Override it with `-DMDIC_COLMAP_SOURCE_DIR=/path/to/colmap` when
  testing another COLMAP source tree. This is the release target for wheels
  because users should not have to install `colmap` separately.
- The bundled upstream tree also carries the PoseLib and faiss sources used by
  COLMAP's FetchContent configuration, so the default build does not need to
  download them at configure time.
- External fallback mode keeps the old command runner for development only. It
  requires `allow_external_executable: true` in the Python options.

The COLMAP source tree under `upstream/` is treated as upstream code. Keep
project-specific logic in this directory's adapter instead of editing COLMAP
internals.
