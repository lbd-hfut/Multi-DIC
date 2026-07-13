// Compact CPU feature frontend derived from COLMAP's SIFT/matcher data flow.
#pragma once

#include "sfm_types.h"

#include <pybind11/pybind11.h>

#include <string>
#include <utility>
#include <vector>

namespace mdic::sfm {

FeatureSet ExtractSift(const pybind11::object& cv2,
                       const std::string& image_root,
                       const std::string& image_name,
                       int max_features,
                       int first_octave,
                       int grid_columns,
                       int grid_rows,
                       int candidate_multiplier);

std::vector<std::pair<int, int>> CreateImagePairs(std::size_t num_images,
                                                   const std::string& matcher,
                                                   int window,
                                                   bool wrap);

std::vector<FeatureMatch> MatchSift(const pybind11::object& cv2,
                                    const FeatureSet& image1,
                                    const FeatureSet& image2,
                                    bool cross_check,
                                    double max_ratio);

}  // namespace mdic::sfm
