// Copyright (c) 2026 Multi-DIC contributors.
//
// Algorithm provenance:
//   COLMAP src/colmap/controllers/incremental_pipeline.cc,
//   src/colmap/sfm/incremental_mapper.cc, incremental_triangulator.cc, and
//   src/colmap/scene/correspondence_graph.cc.
//
// This is a deliberately narrow adaptation for Multi-DIC: CPU SIFT, selected
// pair graphs, calibrated two-view initialization, next-image PnP, track
// triangulation, and local pose/point refinement. It is not a general COLMAP
// replacement and does not carry database, rig, MVS, GUI, or retrieval code.
#include "incremental_sfm.h"

#include "colmap_sparse_mapper.h"
#include "feature.h"
#include "geometry.h"
#include "model_io.h"
#include "sfm_types.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mdic::sfm {
namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

int OptionInt(const py::dict& options, const char* key, const int fallback) {
  const py::str name(key);
  return options.contains(name) && !options[name].is_none() ? options[name].cast<int>() : fallback;
}

double OptionDouble(const py::dict& options, const char* key, const double fallback) {
  const py::str name(key);
  return options.contains(name) && !options[name].is_none() ? options[name].cast<double>() : fallback;
}

bool OptionBool(const py::dict& options, const char* key, const bool fallback) {
  const py::str name(key);
  return options.contains(name) && !options[name].is_none() ? options[name].cast<bool>() : fallback;
}

std::string OptionString(const py::dict& options, const char* key, const std::string& fallback) {
  const py::str name(key);
  return options.contains(name) && !options[name].is_none() ? py::str(options[name]).cast<std::string>() : fallback;
}

class DisjointSet {
 public:
  explicit DisjointSet(const std::size_t size) : parent_(size), rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  std::size_t Find(const std::size_t value) {
    if (parent_[value] != value) {
      parent_[value] = Find(parent_[value]);
    }
    return parent_[value];
  }

  void Union(const std::size_t lhs, const std::size_t rhs) {
    std::size_t root_lhs = Find(lhs);
    std::size_t root_rhs = Find(rhs);
    if (root_lhs == root_rhs) {
      return;
    }
    if (rank_[root_lhs] < rank_[root_rhs]) {
      std::swap(root_lhs, root_rhs);
    }
    parent_[root_rhs] = root_lhs;
    if (rank_[root_lhs] == rank_[root_rhs]) {
      ++rank_[root_lhs];
    }
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<int> rank_;
};

std::vector<std::size_t> FeatureOffsets(const std::vector<FeatureSet>& images) {
  std::vector<std::size_t> offsets(images.size() + 1, 0);
  for (std::size_t idx = 0; idx < images.size(); ++idx) {
    offsets[idx + 1] = offsets[idx] + images[idx].points.size();
  }
  return offsets;
}

void BuildTracks(const std::vector<ImagePair>& pairs, Reconstruction* reconstruction) {
  const std::vector<std::size_t> offsets = FeatureOffsets(reconstruction->images);
  DisjointSet graph(offsets.back());
  for (const ImagePair& pair : pairs) {
    for (const FeatureMatch& match : pair.matches) {
      graph.Union(offsets[static_cast<std::size_t>(pair.image1)] + static_cast<std::size_t>(match.feature1),
                  offsets[static_cast<std::size_t>(pair.image2)] + static_cast<std::size_t>(match.feature2));
    }
  }

  std::unordered_map<std::size_t, std::map<int, int>> components;
  for (std::size_t image_idx = 0; image_idx < reconstruction->images.size(); ++image_idx) {
    for (std::size_t feature_idx = 0; feature_idx < reconstruction->images[image_idx].points.size(); ++feature_idx) {
      const std::size_t node = offsets[image_idx] + feature_idx;
      const std::size_t root = graph.Find(node);
      // A valid COLMAP track has at most one feature per image. Ratio and
      // uniqueness matching make conflicts rare; retain the first when a
      // transitive mismatch would otherwise create an invalid track.
      components[root].emplace(static_cast<int>(image_idx), static_cast<int>(feature_idx));
    }
  }

  reconstruction->feature_to_track.resize(reconstruction->images.size());
  for (std::size_t image_idx = 0; image_idx < reconstruction->images.size(); ++image_idx) {
    reconstruction->feature_to_track[image_idx].assign(reconstruction->images[image_idx].points.size(), -1);
  }
  for (const auto& [unused_root, observations] : components) {
    (void)unused_root;
    if (observations.size() < 2) {
      continue;
    }
    Track track;
    for (const auto& [image, feature] : observations) {
      track.observations.push_back({image, feature});
    }
    const int track_idx = static_cast<int>(reconstruction->tracks.size());
    reconstruction->tracks.push_back(std::move(track));
    for (const Observation& observation : reconstruction->tracks.back().observations) {
      reconstruction->feature_to_track[static_cast<std::size_t>(observation.image)]
                                      [static_cast<std::size_t>(observation.feature)] = track_idx;
    }
  }
}

bool ObservationForImage(const Track& track, const int image, int* feature) {
  for (const Observation& observation : track.observations) {
    if (observation.image == image) {
      *feature = observation.feature;
      return true;
    }
  }
  return false;
}

std::size_t TriangulateTracks(const py::object& cv2,
                              const double max_error,
                              Reconstruction* reconstruction) {
  std::size_t added = 0;
  for (Track& track : reconstruction->tracks) {
    double best_baseline = -1.0;
    const Observation* best1 = nullptr;
    const Observation* best2 = nullptr;
    for (std::size_t i = 0; i < track.observations.size(); ++i) {
      const Observation& observation1 = track.observations[i];
      if (!reconstruction->poses[static_cast<std::size_t>(observation1.image)].registered) {
        continue;
      }
      for (std::size_t j = i + 1; j < track.observations.size(); ++j) {
        const Observation& observation2 = track.observations[j];
        if (!reconstruction->poses[static_cast<std::size_t>(observation2.image)].registered) {
          continue;
        }
        const double baseline = Distance(
            CameraCenter(reconstruction->poses[static_cast<std::size_t>(observation1.image)]),
            CameraCenter(reconstruction->poses[static_cast<std::size_t>(observation2.image)]));
        if (baseline > best_baseline) {
          best_baseline = baseline;
          best1 = &observation1;
          best2 = &observation2;
        }
      }
    }
    if (best1 == nullptr || best2 == nullptr || best_baseline < 1e-8) {
      continue;
    }
    Vec3 xyz;
    double error = 0.0;
    const Vec2 point1 = reconstruction->images[static_cast<std::size_t>(best1->image)]
                            .points[static_cast<std::size_t>(best1->feature)];
    const Vec2 point2 = reconstruction->images[static_cast<std::size_t>(best2->image)]
                            .points[static_cast<std::size_t>(best2->feature)];
    if (Triangulate(cv2,
                    reconstruction->cameras[static_cast<std::size_t>(best1->image)],
                    reconstruction->cameras[static_cast<std::size_t>(best2->image)],
                    reconstruction->poses[static_cast<std::size_t>(best1->image)],
                    reconstruction->poses[static_cast<std::size_t>(best2->image)],
                    point1,
                    point2,
                    max_error,
                    &xyz,
                    &error)) {
      added += track.triangulated ? 0U : 1U;
      track.xyz = xyz;
      track.error = error;
      track.triangulated = true;
    }
  }
  return added;
}

std::size_t CollectPoseCorrespondences(const Reconstruction& reconstruction,
                                       const int image,
                                       std::vector<Vec3>* points3d,
                                       std::vector<Vec2>* points2d) {
  points3d->clear();
  points2d->clear();
  for (const Track& track : reconstruction.tracks) {
    if (!track.triangulated) {
      continue;
    }
    int feature = -1;
    if (ObservationForImage(track, image, &feature)) {
      points3d->push_back(track.xyz);
      points2d->push_back(reconstruction.images[static_cast<std::size_t>(image)]
                              .points[static_cast<std::size_t>(feature)]);
    }
  }
  return points3d->size();
}

bool RegisterBestImage(const py::object& cv2,
                       const int min_inliers,
                       const double max_error,
                       Reconstruction* reconstruction) {
  int best_image = -1;
  std::size_t best_correspondences = 0;
  std::vector<Vec3> points3d;
  std::vector<Vec2> points2d;
  for (std::size_t image = 0; image < reconstruction->images.size(); ++image) {
    if (reconstruction->poses[image].registered) {
      continue;
    }
    const std::size_t count = CollectPoseCorrespondences(
        *reconstruction, static_cast<int>(image), &points3d, &points2d);
    if (count > best_correspondences) {
      best_correspondences = count;
      best_image = static_cast<int>(image);
    }
  }
  if (best_image < 0 || best_correspondences < static_cast<std::size_t>(std::max(6, min_inliers))) {
    return false;
  }
  CollectPoseCorrespondences(*reconstruction, best_image, &points3d, &points2d);
  Pose pose;
  if (!EstimateAbsolutePose(cv2,
                            reconstruction->cameras[static_cast<std::size_t>(best_image)],
                            points3d,
                            points2d,
                            max_error,
                            std::min<int>(min_inliers, static_cast<int>(points3d.size())),
                            &pose)) {
    return false;
  }
  reconstruction->poses[static_cast<std::size_t>(best_image)] = pose;
  return true;
}

Mat3 Transpose(const Mat3& matrix) {
  return {matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]};
}

Vec3 Multiply(const Mat3& matrix, const Vec3& value) {
  return {matrix[0] * value.x + matrix[1] * value.y + matrix[2] * value.z,
          matrix[3] * value.x + matrix[4] * value.y + matrix[5] * value.z,
          matrix[6] * value.x + matrix[7] * value.y + matrix[8] * value.z};
}

bool RegisterByRelativeFallback(const std::vector<ImagePair>& pairs, Reconstruction* reconstruction) {
  const ImagePair* best = nullptr;
  int target = -1;
  bool reverse = false;
  for (const ImagePair& pair : pairs) {
    const bool first_registered = reconstruction->poses[static_cast<std::size_t>(pair.image1)].registered;
    const bool second_registered = reconstruction->poses[static_cast<std::size_t>(pair.image2)].registered;
    if (first_registered == second_registered) {
      continue;
    }
    if (best == nullptr || pair.matches.size() > best->matches.size()) {
      best = &pair;
      target = first_registered ? pair.image2 : pair.image1;
      reverse = !first_registered;
    }
  }
  if (best == nullptr) {
    return false;
  }
  if (!reverse) {
    reconstruction->poses[static_cast<std::size_t>(target)] = ComposeRelativePose(
        reconstruction->poses[static_cast<std::size_t>(best->image1)],
        best->relative_rotation,
        best->relative_translation);
  } else {
    const Mat3 inverse_rotation = Transpose(best->relative_rotation);
    const Vec3 rotated = Multiply(inverse_rotation, best->relative_translation);
    const Vec3 inverse_translation{-rotated.x, -rotated.y, -rotated.z};
    reconstruction->poses[static_cast<std::size_t>(target)] = ComposeRelativePose(
        reconstruction->poses[static_cast<std::size_t>(best->image2)], inverse_rotation, inverse_translation);
  }
  return true;
}

void RefineRegisteredPoses(const py::object& cv2,
                           const int initial_image1,
                           const int initial_image2,
                           const int min_inliers,
                           const double max_error,
                           Reconstruction* reconstruction) {
  std::vector<Vec3> points3d;
  std::vector<Vec2> points2d;
  for (std::size_t image = 0; image < reconstruction->images.size(); ++image) {
    if (!reconstruction->poses[image].registered || static_cast<int>(image) == initial_image1 ||
        static_cast<int>(image) == initial_image2) {
      continue;
    }
    CollectPoseCorrespondences(*reconstruction, static_cast<int>(image), &points3d, &points2d);
    if (points3d.size() < 6) {
      continue;
    }
    Pose refined;
    if (EstimateAbsolutePose(cv2,
                             reconstruction->cameras[image],
                             points3d,
                             points2d,
                             max_error,
                             std::min<int>(min_inliers, static_cast<int>(points3d.size())),
                             &refined)) {
      reconstruction->poses[image] = refined;
    }
  }
}

std::size_t NumRegisteredImages(const Reconstruction& reconstruction) {
  return static_cast<std::size_t>(std::count_if(
      reconstruction.poses.begin(), reconstruction.poses.end(), [](const Pose& pose) { return pose.registered; }));
}

std::size_t NumTriangulatedPoints(const Reconstruction& reconstruction) {
  return static_cast<std::size_t>(std::count_if(reconstruction.tracks.begin(), reconstruction.tracks.end(),
                                                [](const Track& track) { return track.triangulated; }));
}

py::dict Step(const std::string& name, const py::dict& details = py::dict()) {
  py::dict result(details);
  result["name"] = name;
  result["backend"] = "embedded_colmap_sparse_source";
  return result;
}

}  // namespace

py::dict RunCpuSfm(const std::string& database_path,
                   const std::string& image_path,
                   const std::string& sparse_path,
                   const std::string& text_path,
                   const std::vector<std::string>& image_names,
                   const py::dict& options) {
  if (image_names.size() < 2) {
    throw std::runtime_error("Native COLMAP SfM requires at least two images.");
  }
  const py::object cv2 = py::module_::import("cv2");
  cv2.attr("setRNGSeed")(OptionInt(options, "random_seed", 0));
  if (OptionString(options, "camera_model", "SIMPLE_RADIAL") != "SIMPLE_RADIAL") {
    throw std::runtime_error("Native COLMAP SfM currently supports camera_model=SIMPLE_RADIAL only.");
  }

  const int max_features = OptionInt(options, "max_features", 8192);
  const int first_octave = OptionInt(options, "first_octave", -1);
  const int feature_grid_columns = OptionInt(options, "feature_grid_columns", 12);
  const int feature_grid_rows = OptionInt(options, "feature_grid_rows", 9);
  const int feature_candidate_multiplier = OptionInt(options, "feature_candidate_multiplier", 4);
  const int min_num_matches = OptionInt(options, "min_num_matches", 8);
  const double init_max_error = OptionDouble(options, "init_max_error", 4.0);
  const double max_ratio = OptionDouble(options, "max_ratio", 0.8);
  const double initial_focal_factor = OptionDouble(options, "initial_focal_length_factor", 1.2);
  const std::string matcher_name = OptionString(options, "matcher", "exhaustive");
  const int matching_window = OptionInt(options, "matching_window", 2);
  const bool wrap_matching = OptionBool(options, "wrap_matching", true);
  const bool cross_check = OptionBool(options, "cross_check", false);

  Reconstruction reconstruction;
  reconstruction.images.reserve(image_names.size());
  for (const std::string& image_name : image_names) {
    reconstruction.images.push_back(
        ExtractSift(cv2,
                    image_path,
                    image_name,
                    max_features,
                    first_octave,
                    feature_grid_columns,
                    feature_grid_rows,
                    feature_candidate_multiplier));
    const FeatureSet& image = reconstruction.images.back();
    reconstruction.cameras.push_back(
        {initial_focal_factor * std::max(image.width, image.height),
         0.5 * image.width,
         0.5 * image.height,
         0.0,
         image.width,
         image.height});
  }
  reconstruction.poses.resize(image_names.size());

  py::list steps;
  py::dict feature_details;
  py::list feature_counts;
  py::list feature_lower_half_fractions;
  py::list feature_vertical_spans;
  for (const FeatureSet& image : reconstruction.images) {
    feature_counts.append(image.points.size());
    std::size_t lower_half = 0;
    double min_y = static_cast<double>(image.height);
    double max_y = 0.0;
    for (const Vec2& point : image.points) {
      lower_half += point.y >= 0.5 * image.height ? 1 : 0;
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
    feature_lower_half_fractions.append(
        image.points.empty() ? 0.0 : static_cast<double>(lower_half) / image.points.size());
    feature_vertical_spans.append(
        image.points.empty() ? 0.0 : (max_y - min_y) / std::max(1, image.height));
  }
  feature_details["feature_counts"] = feature_counts;
  feature_details["lower_half_fractions"] = feature_lower_half_fractions;
  feature_details["vertical_spans"] = feature_vertical_spans;
  steps.append(Step("feature_extraction", feature_details));

  std::vector<ImagePair> verified_pairs;
  const auto requested_pairs = CreateImagePairs(image_names.size(), matcher_name, matching_window, wrap_matching);
  for (const auto& [image1, image2] : requested_pairs) {
    ImagePair pair;
    pair.image1 = image1;
    pair.image2 = image2;
    pair.matches = MatchSift(cv2,
                             reconstruction.images[static_cast<std::size_t>(image1)],
                             reconstruction.images[static_cast<std::size_t>(image2)],
                             cross_check,
                             max_ratio);
    if (pair.matches.size() < static_cast<std::size_t>(min_num_matches)) {
      continue;
    }
    if (EstimateTwoViewGeometry(cv2,
                                reconstruction.images[static_cast<std::size_t>(image1)],
                                reconstruction.images[static_cast<std::size_t>(image2)],
                                reconstruction.cameras[static_cast<std::size_t>(image1)],
                                reconstruction.cameras[static_cast<std::size_t>(image2)],
                                init_max_error,
                                min_num_matches,
                                &pair.matches,
                                &pair.relative_rotation,
                                &pair.relative_translation)) {
      verified_pairs.push_back(std::move(pair));
    }
  }
  if (verified_pairs.empty()) {
    throw std::runtime_error("No image pair passed compact SfM geometric verification.");
  }
  py::dict matching_details;
  matching_details["requested_pairs"] = requested_pairs.size();
  matching_details["verified_pairs"] = verified_pairs.size();
  py::list pair_match_counts;
  std::vector<std::size_t> matches_per_image(image_names.size(), 0);
  for (const ImagePair& pair : verified_pairs) {
    py::dict item;
    item["image1"] = image_names[static_cast<std::size_t>(pair.image1)];
    item["image2"] = image_names[static_cast<std::size_t>(pair.image2)];
    item["inlier_matches"] = pair.matches.size();
    pair_match_counts.append(item);
    matches_per_image[static_cast<std::size_t>(pair.image1)] += pair.matches.size();
    matches_per_image[static_cast<std::size_t>(pair.image2)] += pair.matches.size();
  }
  py::list image_match_counts;
  for (std::size_t index = 0; index < image_names.size(); ++index) {
    py::dict item;
    item["image"] = image_names[index];
    item["verified_inlier_incidence"] = matches_per_image[index];
    image_match_counts.append(item);
  }
  matching_details["pair_match_counts"] = pair_match_counts;
  matching_details["image_match_counts"] = image_match_counts;
  steps.append(Step("feature_matching_and_two_view_geometry", matching_details));

  py::dict mapping_details;
  if (!RunColmapSparseMapper(verified_pairs, options, &reconstruction, &mapping_details)) {
    throw std::runtime_error("COLMAP sparse source mapper failed to produce a usable reconstruction.");
  }
  const std::size_t registered_images = NumRegisteredImages(reconstruction);
  const std::size_t triangulated_points = NumTriangulatedPoints(reconstruction);
  if (registered_images < 2 || triangulated_points == 0) {
    throw std::runtime_error("COLMAP sparse source mapper did not produce a usable sparse reconstruction.");
  }

  const fs::path text_model = fs::path(text_path) / "0";
  const fs::path sparse_model = fs::path(sparse_path) / "0";
  WriteTextModel(reconstruction, text_model);
  WriteTextModel(reconstruction, sparse_model);

  fs::create_directories(fs::path(database_path).parent_path());
  std::ofstream manifest(fs::path(database_path).parent_path() / "compact_sfm_manifest.txt");
  manifest << "backend=embedded_colmap_sparse_source\n"
           << "database=not-created; correspondences are in-memory\n"
           << "registered_images=" << registered_images << "\n"
           << "points3D=" << triangulated_points << "\n";

  mapping_details["registered_images_after_export"] = registered_images;
  mapping_details["points3D_after_export"] = triangulated_points;
  steps.append(Step("colmap_incremental_mapping", mapping_details));
  steps.append(Step("write_text_model"));

  py::dict result;
  result["backend"] = "embedded_colmap_sparse_source";
  result["model_ids"] = py::make_tuple("0");
  result["database_path"] = database_path;
  result["sparse_path"] = sparse_path;
  result["text_path"] = text_path;
  result["steps"] = steps;
  result["command_logs"] = py::list();
  return result;
}

py::dict Capabilities() {
  py::dict result;
  result["embedded_colmap"] = true;
  result["implementation"] = "colmap_sparse_source_port";
  result["cpu_only"] = true;
  result["feature_extraction"] = "opencv_cpu_sift_colmap_parameters";
  result["matching"] = "ring/sequential/exhaustive";
  result["mapping"] = "colmap_correspondence_graph_incremental_mapper";
  result["refinement"] = "colmap_local_and_global_ceres_bundle_adjustment";
  result["exports"] = py::make_tuple("text_colmap");
  result["excluded"] = py::make_tuple("cuda", "gui", "dense_mvs", "meshing",
                                      "retrieval", "openimageio_bitmap_io");
  return result;
}

}  // namespace mdic::sfm
