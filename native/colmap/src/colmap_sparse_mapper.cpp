// Copyright (c) 2026 Multi-DIC contributors.
//
// This file intentionally contains only glue code. The SfM logic is delegated
// to COLMAP's CorrespondenceGraph, IncrementalMapper, IncrementalTriangulator,
// ObservationManager, and Ceres bundle adjustment sources linked into the
// native extension.
#include "colmap_sparse_mapper.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/feature/types.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/rig.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/incremental_triangulator.h"
#include "colmap/util/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mdic::sfm {
namespace py = pybind11;

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

colmap::FeatureMatches ToColmapMatches(const ImagePair& pair) {
  colmap::FeatureMatches matches;
  matches.reserve(pair.matches.size());
  for (const FeatureMatch& match : pair.matches) {
    if (match.feature1 >= 0 && match.feature2 >= 0) {
      matches.emplace_back(static_cast<colmap::point2D_t>(match.feature1),
                           static_cast<colmap::point2D_t>(match.feature2));
    }
  }
  return matches;
}

std::shared_ptr<colmap::DatabaseCache> BuildDatabaseCache(
    const std::vector<ImagePair>& verified_pairs,
    const Reconstruction& reconstruction) {
  auto cache = std::make_shared<colmap::DatabaseCache>();

  for (std::size_t idx = 0; idx < reconstruction.images.size(); ++idx) {
    const colmap::image_t image_id = static_cast<colmap::image_t>(idx + 1);
    const colmap::camera_t camera_id = image_id;
    const CameraModel& src_camera = reconstruction.cameras[idx];
    colmap::Camera camera = colmap::Camera::CreateFromModelName(
        camera_id,
        "SIMPLE_RADIAL",
        src_camera.focal,
        static_cast<std::size_t>(src_camera.width),
        static_cast<std::size_t>(src_camera.height));
    camera.params[0] = src_camera.focal;
    camera.params[1] = src_camera.cx;
    camera.params[2] = src_camera.cy;
    camera.params[3] = src_camera.radial;
    camera.has_prior_focal_length = true;
    cache->AddCamera(camera);

    colmap::Rig rig;
    rig.SetRigId(camera_id);
    rig.AddRefSensor(camera.SensorId());
    cache->AddRig(std::move(rig));

    colmap::Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera_id);
    image.SetName(reconstruction.images[idx].name);
    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(reconstruction.images[idx].points.size());
    for (const Vec2& point : reconstruction.images[idx].points) {
      points2D.emplace_back(point.x, point.y);
    }
    image.SetPoints2D(std::move(points2D));

    colmap::Frame frame;
    frame.SetFrameId(image_id);
    frame.SetRigId(camera_id);
    frame.AddDataId(image.DataId());
    image.SetFrameId(frame.FrameId());
    cache->AddFrame(std::move(frame));
    cache->AddImage(std::move(image));
  }

  for (const ImagePair& pair : verified_pairs) {
    colmap::TwoViewGeometry tvg;
    tvg.config = colmap::TwoViewGeometry::CALIBRATED;
    tvg.inlier_matches = ToColmapMatches(pair);
    if (!tvg.inlier_matches.empty()) {
      cache->CorrespondenceGraph()->AddTwoViewGeometry(
          static_cast<colmap::image_t>(pair.image1 + 1),
          static_cast<colmap::image_t>(pair.image2 + 1),
          std::move(tvg));
    }
  }
  cache->CorrespondenceGraph()->Finalize();
  return cache;
}

void ConvertFromColmap(const colmap::Reconstruction& src,
                       Reconstruction* dst) {
  dst->poses.assign(dst->images.size(), Pose{});
  dst->tracks.clear();
  dst->feature_to_track.assign(dst->images.size(), {});
  for (std::size_t image_idx = 0; image_idx < dst->images.size(); ++image_idx) {
    dst->feature_to_track[image_idx].assign(dst->images[image_idx].points.size(), -1);
  }

  for (const auto& [image_id, image] : src.Images()) {
    if (image_id == colmap::kInvalidImageId || image_id == 0 ||
        image_id > dst->images.size() || !image.HasPose()) {
      continue;
    }
    const std::size_t idx = static_cast<std::size_t>(image_id - 1);
    const colmap::Rigid3d cam_from_world = image.CamFromWorld();
    const Eigen::Matrix3d rot = cam_from_world.rotation().toRotationMatrix();
    Pose& pose = dst->poses[idx];
    pose.registered = true;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        pose.rotation[static_cast<std::size_t>(r * 3 + c)] = rot(r, c);
      }
    }
    pose.translation.x = cam_from_world.translation().x();
    pose.translation.y = cam_from_world.translation().y();
    pose.translation.z = cam_from_world.translation().z();

    const colmap::Camera& camera = src.Camera(image.CameraId());
    CameraModel& dst_camera = dst->cameras[idx];
    dst_camera.focal = camera.FocalLength();
    dst_camera.cx = camera.PrincipalPointX();
    dst_camera.cy = camera.PrincipalPointY();
    if (camera.ExtraParamsIdxs().size() > 0) {
      dst_camera.radial = camera.params[camera.ExtraParamsIdxs()[0]];
    }
  }

  for (const auto& [unused_point_id, point3D] : src.Points3D()) {
    (void)unused_point_id;
    Track track;
    track.xyz = {point3D.xyz.x(), point3D.xyz.y(), point3D.xyz.z()};
    track.error = point3D.error;
    track.triangulated = true;
    for (const colmap::TrackElement& element : point3D.track.Elements()) {
      if (element.image_id == 0 || element.image_id > dst->images.size()) {
        continue;
      }
      const int image = static_cast<int>(element.image_id - 1);
      const int feature = static_cast<int>(element.point2D_idx);
      if (feature >= 0 &&
          static_cast<std::size_t>(feature) < dst->feature_to_track[static_cast<std::size_t>(image)].size()) {
        track.observations.push_back({image, feature});
      }
    }
    if (track.observations.size() < 2) {
      continue;
    }
    const int track_idx = static_cast<int>(dst->tracks.size());
    for (const Observation& obs : track.observations) {
      dst->feature_to_track[static_cast<std::size_t>(obs.image)]
                           [static_cast<std::size_t>(obs.feature)] = track_idx;
    }
    dst->tracks.push_back(std::move(track));
  }
}

py::list ComputeVisiblePoint3DDiagnostics(
    const colmap::Reconstruction& reconstruction,
    const colmap::CorrespondenceGraph& correspondence_graph,
    const std::size_t num_images) {
  py::list diagnostics;
  std::vector<colmap::CorrespondenceGraph::Correspondence> correspondences;
  for (std::size_t image_idx = 0; image_idx < num_images; ++image_idx) {
    const colmap::image_t image_id = static_cast<colmap::image_t>(image_idx + 1);
    const colmap::Image& image = reconstruction.Image(image_id);
    std::unordered_set<colmap::point3D_t> visible_point3D_ids;
    colmap::point2D_t visible_observations = 0;
    for (colmap::point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
      correspondences.clear();
      correspondence_graph.ExtractTransitiveCorrespondences(
          image_id, point2D_idx, 100, &correspondences);
      bool sees_point3D = false;
      for (const auto& correspondence : correspondences) {
        if (!reconstruction.ExistsImage(correspondence.image_id)) {
          continue;
        }
        const colmap::Image& corr_image = reconstruction.Image(correspondence.image_id);
        if (!corr_image.HasPose()) {
          continue;
        }
        const colmap::Point2D& corr_point =
            corr_image.Point2D(correspondence.point2D_idx);
        if (corr_point.HasPoint3D()) {
          visible_point3D_ids.insert(corr_point.point3D_id);
          sees_point3D = true;
        }
      }
      if (sees_point3D) {
        ++visible_observations;
      }
    }
    py::dict item;
    item["image"] = image.Name();
    item["registered"] = image.HasPose();
    item["visible_observations"] = visible_observations;
    item["visible_points3D"] = visible_point3D_ids.size();
    diagnostics.append(std::move(item));
  }
  return diagnostics;
}

struct ModelQuality {
  std::size_t registered_images = 0;
  std::size_t points3D = 0;
  std::vector<std::size_t> observations;
  std::vector<double> coverage;
  std::size_t min_observations = 0;
  double min_coverage = 0.0;
  double mean_reprojection_error = std::numeric_limits<double>::infinity();
  double max_center_radius_ratio = std::numeric_limits<double>::infinity();
  double max_center_gap_ratio = std::numeric_limits<double>::infinity();
  bool all_images_registered = false;
  bool observations_acceptable = false;
  bool centers_acceptable = false;
};

Vec3 CameraCenter(const Pose& pose) {
  return {
      -(pose.rotation[0] * pose.translation.x + pose.rotation[3] * pose.translation.y +
        pose.rotation[6] * pose.translation.z),
      -(pose.rotation[1] * pose.translation.x + pose.rotation[4] * pose.translation.y +
        pose.rotation[7] * pose.translation.z),
      -(pose.rotation[2] * pose.translation.x + pose.rotation[5] * pose.translation.y +
        pose.rotation[8] * pose.translation.z)};
}

double Distance(const Vec3& a, const Vec3& b) {
  return std::sqrt((a.x - b.x) * (a.x - b.x) +
                   (a.y - b.y) * (a.y - b.y) +
                   (a.z - b.z) * (a.z - b.z));
}

double Median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 != 0) return upper;
  std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
  return 0.5 * (upper + values[middle - 1]);
}

ModelQuality EvaluateModel(const Reconstruction& reconstruction) {
  ModelQuality quality;
  quality.observations.assign(reconstruction.images.size(), 0);
  quality.coverage.assign(reconstruction.images.size(), 0.0);
  std::vector<double> errors;
  std::vector<std::vector<Vec2>> observed_points(reconstruction.images.size());
  for (const Track& track : reconstruction.tracks) {
    if (!track.triangulated) continue;
    ++quality.points3D;
    if (std::isfinite(track.error)) errors.push_back(track.error);
    for (const Observation& observation : track.observations) {
      const std::size_t image = static_cast<std::size_t>(observation.image);
      if (image >= reconstruction.poses.size() || !reconstruction.poses[image].registered ||
          observation.feature < 0 ||
          static_cast<std::size_t>(observation.feature) >= reconstruction.images[image].points.size()) {
        continue;
      }
      ++quality.observations[image];
      observed_points[image].push_back(
          reconstruction.images[image].points[static_cast<std::size_t>(observation.feature)]);
    }
  }
  quality.mean_reprojection_error = errors.empty()
      ? std::numeric_limits<double>::infinity()
      : std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

  quality.min_observations = std::numeric_limits<std::size_t>::max();
  quality.min_coverage = 1.0;
  std::vector<Vec3> centers;
  for (std::size_t image = 0; image < reconstruction.images.size(); ++image) {
    if (!reconstruction.poses[image].registered) {
      quality.min_observations = 0;
      quality.min_coverage = 0.0;
      continue;
    }
    ++quality.registered_images;
    centers.push_back(CameraCenter(reconstruction.poses[image]));
    quality.min_observations = std::min(quality.min_observations, quality.observations[image]);
    if (observed_points[image].size() >= 3) {
      double min_x = std::numeric_limits<double>::infinity();
      double min_y = std::numeric_limits<double>::infinity();
      double max_x = -std::numeric_limits<double>::infinity();
      double max_y = -std::numeric_limits<double>::infinity();
      for (const Vec2& point : observed_points[image]) {
        min_x = std::min(min_x, point.x); max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y); max_y = std::max(max_y, point.y);
      }
      const FeatureSet& features = reconstruction.images[image];
      quality.coverage[image] = ((max_x - min_x) * (max_y - min_y)) /
          std::max(1.0, static_cast<double>(features.width) * features.height);
    }
    quality.min_coverage = std::min(quality.min_coverage, quality.coverage[image]);
  }
  if (quality.min_observations == std::numeric_limits<std::size_t>::max()) quality.min_observations = 0;

  if (centers.size() >= 3) {
    Vec3 centroid{};
    for (const Vec3& center : centers) {
      centroid.x += center.x; centroid.y += center.y; centroid.z += center.z;
    }
    centroid.x /= centers.size(); centroid.y /= centers.size(); centroid.z /= centers.size();
    std::vector<double> radii;
    std::vector<double> nearest;
    for (std::size_t i = 0; i < centers.size(); ++i) {
      radii.push_back(Distance(centers[i], centroid));
      double nearest_distance = std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < centers.size(); ++j) {
        if (i != j) nearest_distance = std::min(nearest_distance, Distance(centers[i], centers[j]));
      }
      nearest.push_back(nearest_distance);
    }
    const double median_radius = std::max(1e-12, Median(radii));
    const double median_nearest = std::max(1e-12, Median(nearest));
    quality.max_center_radius_ratio = *std::max_element(radii.begin(), radii.end()) / median_radius;
    quality.max_center_gap_ratio = *std::max_element(nearest.begin(), nearest.end()) / median_nearest;
  }
  quality.all_images_registered = quality.registered_images == reconstruction.images.size();
  quality.observations_acceptable = quality.min_observations >= 20 && quality.min_coverage >= 0.02;
  quality.centers_acceptable = quality.max_center_radius_ratio <= 2.5 && quality.max_center_gap_ratio <= 3.0;
  return quality;
}

py::dict QualityReport(const ModelQuality& quality, const Reconstruction& reconstruction) {
  py::dict result;
  result["registered_images"] = quality.registered_images;
  result["points3D"] = quality.points3D;
  result["min_observations"] = quality.min_observations;
  result["min_coverage"] = quality.min_coverage;
  result["mean_reprojection_error"] = quality.mean_reprojection_error;
  result["max_center_radius_ratio"] = quality.max_center_radius_ratio;
  result["max_center_gap_ratio"] = quality.max_center_gap_ratio;
  result["all_images_registered"] = quality.all_images_registered;
  result["observations_acceptable"] = quality.observations_acceptable;
  result["centers_acceptable"] = quality.centers_acceptable;
  py::list per_image;
  for (std::size_t i = 0; i < reconstruction.images.size(); ++i) {
    py::dict item;
    item["image"] = reconstruction.images[i].name;
    item["registered"] = reconstruction.poses[i].registered;
    item["observations"] = quality.observations[i];
    item["coverage"] = quality.coverage[i];
    per_image.append(std::move(item));
  }
  result["per_image"] = std::move(per_image);
  return result;
}

bool BetterModel(const ModelQuality& lhs, const ModelQuality& rhs) {
  const auto lhs_rank = std::make_tuple(lhs.all_images_registered,
                                        lhs.observations_acceptable,
                                        lhs.registered_images,
                                        lhs.min_observations,
                                        lhs.centers_acceptable);
  const auto rhs_rank = std::make_tuple(rhs.all_images_registered,
                                        rhs.observations_acceptable,
                                        rhs.registered_images,
                                        rhs.min_observations,
                                        rhs.centers_acceptable);
  if (lhs_rank != rhs_rank) return lhs_rank > rhs_rank;
  if (std::abs(lhs.max_center_gap_ratio - rhs.max_center_gap_ratio) > 1e-6)
    return lhs.max_center_gap_ratio < rhs.max_center_gap_ratio;
  if (std::abs(lhs.max_center_radius_ratio - rhs.max_center_radius_ratio) > 1e-6)
    return lhs.max_center_radius_ratio < rhs.max_center_radius_ratio;
  if (std::abs(lhs.min_coverage - rhs.min_coverage) > 1e-6) return lhs.min_coverage > rhs.min_coverage;
  if (std::abs(lhs.mean_reprojection_error - rhs.mean_reprojection_error) > 1e-6)
    return lhs.mean_reprojection_error < rhs.mean_reprojection_error;
  return lhs.points3D > rhs.points3D;
}

}  // namespace

bool RunSingleColmapSparseMapper(const std::vector<ImagePair>& verified_pairs,
                                 const py::dict& options,
                                 const int initial_image1,
                                 const int initial_image2,
                                 Reconstruction* reconstruction,
                                 py::dict* report) {
  auto cache = BuildDatabaseCache(verified_pairs, *reconstruction);
  auto colmap_reconstruction = std::make_shared<colmap::Reconstruction>();
  colmap::IncrementalMapper mapper(cache);

  colmap::IncrementalMapper::Options mapper_options;
  mapper_options.init_min_num_inliers = OptionInt(options, "init_min_num_inliers", 100);
  mapper_options.init_max_error = OptionDouble(options, "init_max_error", 4.0);
  mapper_options.abs_pose_min_num_inliers = OptionInt(options, "abs_pose_min_num_inliers", 30);
  mapper_options.abs_pose_max_error = OptionDouble(options, "abs_pose_max_error", 12.0);
  mapper_options.abs_pose_min_inlier_ratio = OptionDouble(options, "abs_pose_min_inlier_ratio", 0.25);
  mapper_options.abs_pose_refine_focal_length = OptionBool(options, "abs_pose_refine_focal_length", true);
  mapper_options.abs_pose_refine_extra_params = OptionBool(options, "abs_pose_refine_extra_params", true);
  mapper_options.filter_max_reproj_error = OptionDouble(options, "filter_max_reproj_error", 4.0);
  mapper_options.ba_local_num_images = OptionInt(options, "ba_local_num_images", 6);
  mapper_options.num_threads = OptionInt(options, "num_threads", -1);
  mapper_options.random_seed = OptionInt(options, "random_seed", -1);

  colmap::IncrementalTriangulator::Options triangulator_options;
  triangulator_options.ignore_two_view_tracks = false;
  triangulator_options.merge_max_reproj_error = mapper_options.filter_max_reproj_error;
  triangulator_options.complete_max_reproj_error = mapper_options.filter_max_reproj_error;
  triangulator_options.random_seed = mapper_options.random_seed;

  colmap::BundleAdjustmentOptions ba_options;
  ba_options.print_summary = false;
  ba_options.refine_focal_length = true;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = true;

  mapper.BeginReconstruction(colmap_reconstruction);
  colmap::image_t image_id1 = colmap::kInvalidImageId;
  colmap::image_t image_id2 = colmap::kInvalidImageId;
  colmap::Rigid3d cam2_from_cam1;
  image_id1 = static_cast<colmap::image_t>(initial_image1 + 1);
  image_id2 = static_cast<colmap::image_t>(initial_image2 + 1);
  const bool found_initial_pair =
      mapper.EstimateInitialTwoViewGeometry(mapper_options, image_id1, image_id2, cam2_from_cam1);
  if (!found_initial_pair) {
    mapper.EndReconstruction(true);
    return false;
  }
  mapper.RegisterInitialImagePair(mapper_options, image_id1, image_id2, cam2_from_cam1);
  mapper.TriangulateImage(triangulator_options, image_id1);
  mapper.TriangulateImage(triangulator_options, image_id2);
  mapper.IterativeLocalRefinement(2, 0.001, mapper_options, ba_options, triangulator_options, image_id1);
  mapper.IterativeLocalRefinement(2, 0.001, mapper_options, ba_options, triangulator_options, image_id2);
  py::list initial_visible_diagnostics;
  py::list first_next_images;
  if (report != nullptr) {
    initial_visible_diagnostics = ComputeVisiblePoint3DDiagnostics(
        *colmap_reconstruction, *cache->CorrespondenceGraph(), reconstruction->images.size());
    for (const colmap::image_t next_image : mapper.FindNextImages(mapper_options)) {
      first_next_images.append(next_image - 1);
    }
  }

  py::list registration_attempts;
  while (colmap_reconstruction->NumRegImages() < reconstruction->images.size()) {
    bool registered_any = false;
    std::vector<colmap::image_t> next_images = mapper.FindNextImages(mapper_options);
    if (next_images.empty()) {
      next_images = mapper.FindNextImages(mapper_options, true);
      if (next_images.empty()) {
        break;
      }
    }
    for (const colmap::image_t next_image_id : next_images) {
      bool used_structure_less = false;
      bool registered = mapper.RegisterNextImage(mapper_options, next_image_id);
      if (!registered) {
        registered = mapper.RegisterNextStructureLessImage(mapper_options, next_image_id);
        used_structure_less = registered;
      }
      if (report != nullptr) {
        py::dict attempt;
        attempt["image"] = reconstruction->images[static_cast<std::size_t>(next_image_id - 1)].name;
        attempt["image_index"] = next_image_id - 1;
        attempt["registered"] = registered;
        attempt["used_structure_less"] = used_structure_less;
        attempt["registered_images_after_attempt"] = colmap_reconstruction->NumRegImages();
        registration_attempts.append(std::move(attempt));
      }
      if (registered) {
        registered_any = true;
        mapper.TriangulateImage(triangulator_options, next_image_id);
        mapper.IterativeLocalRefinement(
            2, 0.001, mapper_options, ba_options, triangulator_options, next_image_id);
      }
      if (colmap_reconstruction->NumRegImages() >= reconstruction->images.size()) {
        break;
      }
    }
    if (!registered_any) {
      break;
    }
  }

  mapper.IterativeGlobalRefinement(
      2, 0.0005, mapper_options, ba_options, triangulator_options, false);

  // A structure-less fallback can place an image near the correct camera ring
  // while leaving it with only a handful of triangulated observations. Once
  // the rest of the model is mature, discard such weak poses and estimate
  // them again using the now-dense 2D-3D correspondence set. This specifically
  // prevents a nominal 12/12 model with an effectively unusable camera.
  py::list weak_pose_reregistration;
  const std::size_t min_final_observations = static_cast<std::size_t>(
      std::max(20, OptionInt(options, "min_final_observations", 100)));
  for (std::size_t image_idx = 0; image_idx < reconstruction->images.size(); ++image_idx) {
    const colmap::image_t weak_image_id = static_cast<colmap::image_t>(image_idx + 1);
    colmap::Image& weak_image = colmap_reconstruction->Image(weak_image_id);
    if (!weak_image.HasPose() || weak_image.NumPoints3D() >= min_final_observations ||
        weak_image_id == image_id1 || weak_image_id == image_id2) {
      continue;
    }
    const std::size_t observations_before = weak_image.NumPoints3D();
    mapper.ObservationManager().DeRegisterFrame(weak_image.FrameId());
    const bool reregistered = mapper.RegisterNextImage(mapper_options, weak_image_id);
    if (reregistered) {
      mapper.TriangulateImage(triangulator_options, weak_image_id);
      mapper.IterativeLocalRefinement(
          3, 0.0005, mapper_options, ba_options, triangulator_options, weak_image_id);
    } else {
      // Preserve the 12-camera model as a diagnostic candidate if mature PnP
      // still cannot recover the weak image.
      mapper.RegisterNextStructureLessImage(mapper_options, weak_image_id);
      mapper.TriangulateImage(triangulator_options, weak_image_id);
    }
    py::dict item;
    item["image"] = reconstruction->images[image_idx].name;
    item["observations_before"] = observations_before;
    item["reregistered_with_pnp"] = reregistered;
    item["observations_after"] = colmap_reconstruction->Image(weak_image_id).NumPoints3D();
    weak_pose_reregistration.append(std::move(item));
  }
  mapper.CompleteAndMergeTracks(triangulator_options);
  mapper.Retriangulate(triangulator_options);
  mapper.IterativeGlobalRefinement(
      3, 0.0005, mapper_options, ba_options, triangulator_options, false);
  mapper.EndReconstruction(false);

  ConvertFromColmap(*colmap_reconstruction, reconstruction);
  if (report != nullptr) {
    (*report)["registered_images"] = colmap_reconstruction->NumRegImages();
    (*report)["points3D"] = colmap_reconstruction->NumPoints3D();
    (*report)["initial_pair"] = py::make_tuple(image_id1 - 1, image_id2 - 1);
    (*report)["forced_initial_pair"] = true;
    (*report)["refinement"] = "colmap_incremental_mapper_local_and_global_ba";
    (*report)["effective_abs_pose_min_num_inliers"] = mapper_options.abs_pose_min_num_inliers;
    (*report)["effective_abs_pose_min_inlier_ratio"] = mapper_options.abs_pose_min_inlier_ratio;
    (*report)["effective_abs_pose_max_error"] = mapper_options.abs_pose_max_error;
    (*report)["effective_filter_max_reproj_error"] = mapper_options.filter_max_reproj_error;
    (*report)["initial_visible_point3D_diagnostics"] = initial_visible_diagnostics;
    (*report)["initial_next_image_candidates"] = first_next_images;
    (*report)["registration_attempts"] = registration_attempts;
    (*report)["weak_pose_reregistration"] = weak_pose_reregistration;
  }
  return colmap_reconstruction->NumRegImages() >= 2 && colmap_reconstruction->NumPoints3D() > 0;
}

bool RunColmapSparseMapper(const std::vector<ImagePair>& verified_pairs,
                           const py::dict& options,
                           Reconstruction* reconstruction,
                           py::dict* report) {
  const std::size_t num_images = reconstruction->images.size();
  if (num_images < 2) return false;

  // Adjacent ring pairs, including wrap-around, are the primary candidates.
  // Keep the configured pair and historically difficult 5/6 and 8/9 pairs at
  // the front without allowing any one pair to lock the entire reconstruction.
  std::vector<std::pair<int, int>> candidates;
  std::unordered_set<std::uint64_t> seen;
  const auto add_candidate = [&](const int image1, const int image2) {
    if (image1 < 0 || image2 < 0 || image1 == image2 ||
        static_cast<std::size_t>(image1) >= num_images ||
        static_cast<std::size_t>(image2) >= num_images) return;
    const int lo = std::min(image1, image2);
    const int hi = std::max(image1, image2);
    const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) |
                              static_cast<std::uint32_t>(hi);
    if (seen.insert(key).second) candidates.emplace_back(image1, image2);
  };
  add_candidate(OptionInt(options, "initial_image1", -1),
                OptionInt(options, "initial_image2", -1));
  add_candidate(5, 6);
  add_candidate(8, 9);
  for (std::size_t image = 0; image < num_images; ++image) {
    add_candidate(static_cast<int>(image), static_cast<int>((image + 1) % num_images));
  }

  Reconstruction best_reconstruction;
  ModelQuality best_quality;
  py::dict best_mapping_report;
  py::list candidate_reports;
  bool have_best = false;
  for (const auto& [image1, image2] : candidates) {
    Reconstruction candidate = *reconstruction;
    candidate.poses.assign(num_images, Pose{});
    candidate.tracks.clear();
    candidate.feature_to_track.clear();
    py::dict candidate_mapping_report;
    py::dict candidate_report;
    candidate_report["initial_pair"] = py::make_tuple(image1, image2);
    try {
      if (!RunSingleColmapSparseMapper(verified_pairs,
                                       options,
                                       image1,
                                       image2,
                                       &candidate,
                                       &candidate_mapping_report)) {
        candidate_report["usable"] = false;
        candidate_report["reason"] = "initialization_or_mapping_failed";
        candidate_reports.append(std::move(candidate_report));
        continue;
      }
      const ModelQuality quality = EvaluateModel(candidate);
      candidate_report["usable"] = true;
      candidate_report["quality"] = QualityReport(quality, candidate);
      if (!have_best || BetterModel(quality, best_quality)) {
        have_best = true;
        best_quality = quality;
        best_reconstruction = std::move(candidate);
        best_mapping_report = candidate_mapping_report;
      }
    } catch (const std::exception& error) {
      candidate_report["usable"] = false;
      candidate_report["reason"] = error.what();
    }
    candidate_reports.append(std::move(candidate_report));
  }
  if (!have_best) {
    if (report != nullptr) (*report)["candidate_models"] = candidate_reports;
    return false;
  }
  *reconstruction = std::move(best_reconstruction);
  if (report != nullptr) {
    *report = best_mapping_report;
    (*report)["candidate_models"] = candidate_reports;
    (*report)["selected_quality"] = QualityReport(best_quality, *reconstruction);
    (*report)["selection_policy"] =
        "all_registered,min_observations,center_outliers,coverage,reprojection_error,points3D";
  }
  return true;
}

}  // namespace mdic::sfm
