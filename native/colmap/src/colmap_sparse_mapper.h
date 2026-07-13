// Adapter from Multi-DIC's OpenCV feature frontend into COLMAP's sparse mapper.
#pragma once

#include "sfm_types.h"

#include <pybind11/pybind11.h>

#include <vector>

namespace mdic::sfm {

bool RunColmapSparseMapper(const std::vector<ImagePair>& verified_pairs,
                           const pybind11::dict& options,
                           Reconstruction* reconstruction,
                           pybind11::dict* report);

}  // namespace mdic::sfm
