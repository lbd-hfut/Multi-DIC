// Copyright (c) 2026 Multi-DIC contributors.
//
// Algorithm provenance:
//   COLMAP src/colmap/feature/sift.cc, matcher.cc, and controllers/pairing.cc.
// The compact backend keeps COLMAP's CPU SIFT -> ratio/uniqueness matching ->
// geometric verification ordering. OpenCV supplies the optimized SIFT and
// brute-force kernels so they do not add a second native dependency graph.
#include "feature.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace mdic::sfm {
namespace py = pybind11;

FeatureSet ExtractSift(const py::object& cv2,
                       const std::string& image_root,
                       const std::string& image_name,
                       const int max_features,
                       const int first_octave,
                       const int grid_columns,
                       const int grid_rows,
                       const int candidate_multiplier) {
  FeatureSet result;
  result.name = image_name;
  result.path = (std::filesystem::path(image_root) / image_name).string();

  const py::object image = cv2.attr("imread")(result.path, cv2.attr("IMREAD_GRAYSCALE"));
  if (image.is_none()) {
    throw std::runtime_error("Could not read SfM image: " + result.path);
  }
  const py::tuple shape = image.attr("shape").cast<py::tuple>();
  result.height = shape[0].cast<int>();
  result.width = shape[1].cast<int>();

  // COLMAP's first_octave=-1 upsamples before SIFT. OpenCV has no direct
  // first-octave switch, so preserve it with a 2x pyramid input and map the
  // returned coordinates back to the original image.
  py::object sift_image = image;
  double coordinate_scale = 1.0;
  if (first_octave < 0) {
    sift_image = cv2.attr("resize")(image,
                                    py::none(),
                                    py::arg("fx") = 2.0,
                                    py::arg("fy") = 2.0,
                                    py::arg("interpolation") = cv2.attr("INTER_LINEAR"));
    coordinate_scale = 0.5;
  }

  // contrastThreshold/3 matches the threshold convention used by COLMAP's
  // VLFeat SIFT wrapper closely enough for the intended CPU frontend.
  // Let OpenCV return the complete detector population. Applying nfeatures
  // here performs a global response truncation which is badly biased on
  // repetitive DIC speckles (typically retaining only the upper image band).
  // The grid quota below is the sole max_features limiter.
  (void)candidate_multiplier;
  const py::object sift = cv2.attr("SIFT_create")(
      py::arg("nfeatures") = 0,
      py::arg("nOctaveLayers") = 3,
      py::arg("contrastThreshold") = 0.006666666666666667,
      py::arg("edgeThreshold") = 10.0,
      py::arg("sigma") = 1.6);
  const py::tuple extracted = sift.attr("detectAndCompute")(sift_image, py::none()).cast<py::tuple>();
  const py::sequence keypoints = extracted[0].cast<py::sequence>();
  if (extracted[1].is_none() || keypoints.empty()) {
    return result;
  }
  const py::tuple sift_shape = sift_image.attr("shape").cast<py::tuple>();
  const double sift_height = sift_shape[0].cast<double>();
  const double sift_width = sift_shape[1].cast<double>();

  struct Candidate {
    std::size_t index = 0;
    double response = 0.0;
  };
  const int columns = std::max(1, grid_columns);
  const int rows = std::max(1, grid_rows);
  const std::size_t num_cells = static_cast<std::size_t>(columns * rows);
  std::vector<std::vector<Candidate>> cells(num_cells);
  for (std::size_t idx = 0; idx < keypoints.size(); ++idx) {
    const py::handle keypoint = keypoints[idx];
    const py::tuple point = keypoint.attr("pt").cast<py::tuple>();
    const double x = point[0].cast<double>();
    const double y = point[1].cast<double>();
    const int column = std::clamp(static_cast<int>(x * columns / sift_width),
                                  0,
                                  columns - 1);
    const int row = std::clamp(static_cast<int>(y * rows / sift_height),
                               0,
                               rows - 1);
    cells[static_cast<std::size_t>(row * columns + column)].push_back(
        {idx, keypoint.attr("response").cast<double>()});
  }
  for (auto& cell : cells) {
    std::sort(cell.begin(), cell.end(), [](const Candidate& lhs, const Candidate& rhs) {
      return lhs.response > rhs.response;
    });
  }

  const std::size_t target = std::min<std::size_t>(static_cast<std::size_t>(std::max(1, max_features)),
                                                   keypoints.size());
  const std::size_t quota = (target + num_cells - 1) / num_cells;
  std::vector<std::size_t> selected;
  selected.reserve(target);
  for (std::size_t rank = 0; rank < quota && selected.size() < target; ++rank) {
    for (const auto& cell : cells) {
      if (rank < cell.size()) {
        selected.push_back(cell[rank].index);
        if (selected.size() == target) {
          break;
        }
      }
    }
  }
  if (selected.size() < target) {
    std::unordered_set<std::size_t> already_selected(selected.begin(), selected.end());
    std::vector<Candidate> remaining;
    for (const auto& cell : cells) {
      for (const Candidate& candidate : cell) {
        if (already_selected.count(candidate.index) == 0) {
          remaining.push_back(candidate);
        }
      }
    }
    std::sort(remaining.begin(), remaining.end(), [](const Candidate& lhs, const Candidate& rhs) {
      return lhs.response > rhs.response;
    });
    for (const Candidate& candidate : remaining) {
      selected.push_back(candidate.index);
      if (selected.size() == target) {
        break;
      }
    }
  }

  result.points.reserve(selected.size());
  for (const std::size_t idx : selected) {
    const py::handle keypoint = keypoints[idx];
    const py::tuple point = keypoint.attr("pt").cast<py::tuple>();
    result.points.push_back({point[0].cast<double>() * coordinate_scale,
                             point[1].cast<double>() * coordinate_scale});
  }
  result.descriptors = py::module_::import("numpy").attr("take")(
      extracted[1], selected, py::arg("axis") = 0);
  return result;
}

std::vector<std::pair<int, int>> CreateImagePairs(const std::size_t num_images,
                                                   const std::string& matcher,
                                                   const int window,
                                                   const bool wrap) {
  std::vector<std::pair<int, int>> result;
  if (num_images < 2) {
    return result;
  }
  if (matcher == "exhaustive") {
    for (std::size_t i = 0; i < num_images; ++i) {
      for (std::size_t j = i + 1; j < num_images; ++j) {
        result.emplace_back(static_cast<int>(i), static_cast<int>(j));
      }
    }
    return result;
  }

  const int count = static_cast<int>(num_images);
  const int effective_window = std::max(1, window);
  for (int i = 0; i < count; ++i) {
    for (int step = 1; step <= effective_window; ++step) {
      int j = i + step;
      if (j >= count) {
        if (!wrap || count <= 2) {
          continue;
        }
        j %= count;
      }
      if (i == j) {
        continue;
      }
      const auto ordered = std::minmax(i, j);
      const std::pair<int, int> pair{ordered.first, ordered.second};
      if (std::find(result.begin(), result.end(), pair) == result.end()) {
        result.push_back(pair);
      }
    }
  }
  return result;
}

std::vector<FeatureMatch> MatchSift(const py::object& cv2,
                                    const FeatureSet& image1,
                                    const FeatureSet& image2,
                                    const bool cross_check,
                                    const double max_ratio) {
  if (image1.descriptors.is_none() || image2.descriptors.is_none() || image1.points.empty() ||
      image2.points.empty()) {
    return {};
  }

  const py::object matcher = cv2.attr("BFMatcher")(cv2.attr("NORM_L2"), false);
  const py::sequence forward =
      matcher.attr("knnMatch")(image1.descriptors, image2.descriptors, 2).cast<py::sequence>();

  std::map<int, FeatureMatch> unique_train;
  for (const py::handle candidates_handle : forward) {
    const py::sequence candidates = py::reinterpret_borrow<py::sequence>(candidates_handle);
    if (candidates.size() < 2) {
      continue;
    }
    const py::object best = candidates[0];
    const py::object second = candidates[1];
    const double distance = best.attr("distance").cast<double>();
    if (distance >= max_ratio * second.attr("distance").cast<double>()) {
      continue;
    }
    FeatureMatch match{best.attr("queryIdx").cast<int>(), best.attr("trainIdx").cast<int>(), distance};
    const auto existing = unique_train.find(match.feature2);
    if (existing == unique_train.end() || match.distance < existing->second.distance) {
      unique_train[match.feature2] = match;
    }
  }

  std::vector<FeatureMatch> matches;
  matches.reserve(unique_train.size());
  if (!cross_check) {
    for (const auto& [unused, match] : unique_train) {
      (void)unused;
      matches.push_back(match);
    }
    return matches;
  }

  const py::sequence reverse =
      matcher.attr("knnMatch")(image2.descriptors, image1.descriptors, 2).cast<py::sequence>();
  std::map<int, int> reverse_best;
  for (const py::handle candidates_handle : reverse) {
    const py::sequence candidates = py::reinterpret_borrow<py::sequence>(candidates_handle);
    if (candidates.size() < 2) {
      continue;
    }
    const py::object best = candidates[0];
    const py::object second = candidates[1];
    if (best.attr("distance").cast<double>() < max_ratio * second.attr("distance").cast<double>()) {
      reverse_best[best.attr("queryIdx").cast<int>()] = best.attr("trainIdx").cast<int>();
    }
  }
  for (const auto& [train, match] : unique_train) {
    const auto reverse_match = reverse_best.find(train);
    if (reverse_match != reverse_best.end() && reverse_match->second == match.feature1) {
      matches.push_back(match);
    }
  }
  return matches;
}

}  // namespace mdic::sfm
