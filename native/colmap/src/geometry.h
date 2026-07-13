// Minimal robust geometry used by the incremental mapper. The operation order
// mirrors COLMAP's two-view initialization, absolute pose and triangulation.
#pragma once

#include "sfm_types.h"

#include <pybind11/pybind11.h>

#include <vector>

namespace mdic::sfm {

bool EstimateTwoViewGeometry(const pybind11::object& cv2,
                             const FeatureSet& image1,
                             const FeatureSet& image2,
                             const CameraModel& camera1,
                             const CameraModel& camera2,
                             double max_error,
                             int min_inliers,
                             std::vector<FeatureMatch>* matches,
                             Mat3* relative_rotation,
                             Vec3* relative_translation);

bool EstimateAbsolutePose(const pybind11::object& cv2,
                          const CameraModel& camera,
                          const std::vector<Vec3>& points3d,
                          const std::vector<Vec2>& points2d,
                          double max_error,
                          int min_inliers,
                          Pose* pose);

bool Triangulate(const pybind11::object& cv2,
                 const CameraModel& camera1,
                 const CameraModel& camera2,
                 const Pose& pose1,
                 const Pose& pose2,
                 const Vec2& point1,
                 const Vec2& point2,
                 double max_error,
                 Vec3* xyz,
                 double* mean_error);

Pose ComposeRelativePose(const Pose& world_to_camera1,
                         const Mat3& camera2_from_camera1,
                         const Vec3& translation);

Vec2 Project(const CameraModel& camera, const Pose& pose, const Vec3& xyz);
Vec3 CameraCenter(const Pose& pose);
double Distance(const Vec3& lhs, const Vec3& rhs);
std::array<double, 4> RotationToQuaternion(const Mat3& rotation);

}  // namespace mdic::sfm
