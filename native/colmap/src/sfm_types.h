// Copyright (c) 2026 Multi-DIC contributors.
//
// This file defines the compact data model used by Multi-DIC's COLMAP-derived
// sparse SfM backend. The reconstruction flow follows COLMAP's incremental
// pipeline and scene model, but intentionally exposes no COLMAP internals.
#pragma once

#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mdic::sfm {

namespace py = pybind11;

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

using Mat3 = std::array<double, 9>;

struct Pose {
  Mat3 rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  Vec3 translation{};
  bool registered = false;
};

struct FeatureSet {
  std::string name;
  std::string path;
  int width = 0;
  int height = 0;
  std::vector<Vec2> points;
  py::object descriptors = py::none();
};

struct FeatureMatch {
  int feature1 = -1;
  int feature2 = -1;
  double distance = 0.0;
};

struct ImagePair {
  int image1 = -1;
  int image2 = -1;
  std::vector<FeatureMatch> matches;
  Mat3 relative_rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  Vec3 relative_translation{};
};

struct Observation {
  int image = -1;
  int feature = -1;
};

struct Track {
  std::vector<Observation> observations;
  Vec3 xyz{};
  double error = 0.0;
  bool triangulated = false;
};

struct CameraModel {
  double focal = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double radial = 0.0;
  int width = 0;
  int height = 0;
};

struct Reconstruction {
  std::vector<FeatureSet> images;
  std::vector<CameraModel> cameras;
  std::vector<Pose> poses;
  std::vector<Track> tracks;
  std::vector<std::vector<int>> feature_to_track;
};

}  // namespace mdic::sfm
