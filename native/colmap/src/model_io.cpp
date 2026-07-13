// Copyright (c) 2026 Multi-DIC contributors.
//
// Text layout follows COLMAP src/colmap/scene/reconstruction_io.cc. Only the
// cameras/images/points3D files consumed by Multi-DIC are emitted.
#include "model_io.h"

#include "geometry.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace mdic::sfm {

void WriteTextModel(const Reconstruction& reconstruction,
                    const std::filesystem::path& model_path) {
  std::filesystem::create_directories(model_path);
  std::ofstream cameras(model_path / "cameras.txt");
  std::ofstream images(model_path / "images.txt");
  std::ofstream points(model_path / "points3D.txt");
  if (!cameras || !images || !points) {
    throw std::runtime_error("Could not create compact COLMAP text model in: " + model_path.string());
  }
  cameras << std::setprecision(17);
  images << std::setprecision(17);
  points << std::setprecision(17);

  std::size_t registered_images = 0;
  for (const Pose& pose : reconstruction.poses) {
    registered_images += pose.registered ? 1U : 0U;
  }
  std::size_t triangulated_points = 0;
  for (const Track& track : reconstruction.tracks) {
    triangulated_points += track.triangulated ? 1U : 0U;
  }

  cameras << "# Camera list with one line of data per camera:\n"
          << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
          << "# Number of cameras: " << registered_images << "\n";
  for (std::size_t image_idx = 0; image_idx < reconstruction.images.size(); ++image_idx) {
    if (!reconstruction.poses[image_idx].registered) {
      continue;
    }
    const CameraModel& camera = reconstruction.cameras[image_idx];
    cameras << image_idx + 1 << " SIMPLE_RADIAL " << camera.width << ' ' << camera.height << ' '
            << camera.focal << ' ' << camera.cx << ' ' << camera.cy << ' ' << camera.radial << '\n';
  }

  images << "# Image list with two lines of data per image:\n"
         << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
         << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
         << "# Number of images: " << registered_images << "\n";
  for (std::size_t image_idx = 0; image_idx < reconstruction.images.size(); ++image_idx) {
    const Pose& pose = reconstruction.poses[image_idx];
    if (!pose.registered) {
      continue;
    }
    const auto quaternion = RotationToQuaternion(pose.rotation);
    images << image_idx + 1 << ' ' << quaternion[0] << ' ' << quaternion[1] << ' ' << quaternion[2] << ' '
           << quaternion[3] << ' ' << pose.translation.x << ' ' << pose.translation.y << ' '
           << pose.translation.z << ' ' << image_idx + 1 << ' ' << reconstruction.images[image_idx].name << '\n';

    const FeatureSet& image = reconstruction.images[image_idx];
    for (std::size_t feature_idx = 0; feature_idx < image.points.size(); ++feature_idx) {
      int point_id = -1;
      if (feature_idx < reconstruction.feature_to_track[image_idx].size()) {
        const int track_idx = reconstruction.feature_to_track[image_idx][feature_idx];
        if (track_idx >= 0 && reconstruction.tracks[static_cast<std::size_t>(track_idx)].triangulated) {
          point_id = track_idx + 1;
        }
      }
      const Vec2& feature = image.points[feature_idx];
      images << feature.x << ' ' << feature.y << ' ' << point_id << ' ';
    }
    images << '\n';
  }

  points << "# 3D point list with one line of data per point:\n"
         << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
         << "# Number of points: " << triangulated_points << "\n";
  for (std::size_t track_idx = 0; track_idx < reconstruction.tracks.size(); ++track_idx) {
    const Track& track = reconstruction.tracks[track_idx];
    if (!track.triangulated) {
      continue;
    }
    points << track_idx + 1 << ' ' << track.xyz.x << ' ' << track.xyz.y << ' ' << track.xyz.z
           << " 128 128 128 " << track.error;
    for (const Observation& observation : track.observations) {
      if (reconstruction.poses[static_cast<std::size_t>(observation.image)].registered) {
        points << ' ' << observation.image + 1 << ' ' << observation.feature;
      }
    }
    points << '\n';
  }

  // COLMAP 4.x readers accept these empty rig/frame files; Multi-DIC does not
  // consume them, but writing valid headers keeps the model interoperable.
  std::ofstream rigs(model_path / "rigs.txt");
  rigs << "# Rig calib list with one line of data per calib:\n# Number of rigs: 0\n";
  std::ofstream frames(model_path / "frames.txt");
  frames << "# Frame list with one line of data per frame:\n# Number of frames: 0\n";
}

}  // namespace mdic::sfm
