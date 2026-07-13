// Copyright (c) 2026 Multi-DIC contributors.
//
// Algorithm provenance:
//   COLMAP src/colmap/estimators/two_view_geometry.cc,
//   src/colmap/estimators/pose.cc, and
//   src/colmap/geometry/triangulation.cc.
// Robust model estimation and cheirality/reprojection filtering are retained;
// OpenCV provides the calibrated five-point, PnP, Rodrigues and SVD kernels.
#include "geometry.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mdic::sfm {
namespace py = pybind11;

namespace {

Mat3 Multiply(const Mat3& lhs, const Mat3& rhs) {
  Mat3 result{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      for (int k = 0; k < 3; ++k) {
        result[row * 3 + col] += lhs[row * 3 + k] * rhs[k * 3 + col];
      }
    }
  }
  return result;
}

Vec3 Multiply(const Mat3& matrix, const Vec3& value) {
  return {matrix[0] * value.x + matrix[1] * value.y + matrix[2] * value.z,
          matrix[3] * value.x + matrix[4] * value.y + matrix[5] * value.z,
          matrix[6] * value.x + matrix[7] * value.y + matrix[8] * value.z};
}

Mat3 Transpose(const Mat3& value) {
  return {value[0], value[3], value[6], value[1], value[4], value[7], value[2], value[5], value[8]};
}

py::array_t<double> CameraMatrix(const CameraModel& camera) {
  py::array_t<double> result(std::vector<py::ssize_t>{3, 3});
  auto matrix = result.mutable_unchecked<2>();
  matrix(0, 0) = camera.focal;
  matrix(0, 1) = 0.0;
  matrix(0, 2) = camera.cx;
  matrix(1, 0) = 0.0;
  matrix(1, 1) = camera.focal;
  matrix(1, 2) = camera.cy;
  matrix(2, 0) = 0.0;
  matrix(2, 1) = 0.0;
  matrix(2, 2) = 1.0;
  return result;
}

py::array_t<double> PointArray(const std::vector<Vec2>& points) {
  py::array_t<double> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(points.size()), 2});
  auto array = result.mutable_unchecked<2>();
  for (py::ssize_t idx = 0; idx < static_cast<py::ssize_t>(points.size()); ++idx) {
    array(idx, 0) = points[static_cast<std::size_t>(idx)].x;
    array(idx, 1) = points[static_cast<std::size_t>(idx)].y;
  }
  return result;
}

Mat3 ReadMat3(const py::handle value) {
  const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || array.size() < 9) {
    throw std::runtime_error("OpenCV returned an invalid 3x3 matrix.");
  }
  const double* data = array.data();
  return {data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]};
}

Vec3 ReadVec3(const py::handle value) {
  const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || array.size() < 3) {
    throw std::runtime_error("OpenCV returned an invalid 3-vector.");
  }
  return {array.data()[0], array.data()[1], array.data()[2]};
}

py::array_t<double> ProjectionMatrix(const CameraModel& camera, const Pose& pose) {
  py::array_t<double> result(std::vector<py::ssize_t>{3, 4});
  auto projection = result.mutable_unchecked<2>();
  const double k[9] = {camera.focal, 0.0, camera.cx,
                       0.0, camera.focal, camera.cy,
                       0.0, 0.0, 1.0};
  const double extrinsic[12] = {
      pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.translation.x,
      pose.rotation[3], pose.rotation[4], pose.rotation[5], pose.translation.y,
      pose.rotation[6], pose.rotation[7], pose.rotation[8], pose.translation.z};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      projection(row, col) = 0.0;
      for (int kidx = 0; kidx < 3; ++kidx) {
        projection(row, col) += k[row * 3 + kidx] * extrinsic[kidx * 4 + col];
      }
    }
  }
  return result;
}

double ReprojectionError(const CameraModel& camera, const Pose& pose, const Vec3& xyz, const Vec2& point) {
  const Vec2 projected = Project(camera, pose, xyz);
  return std::hypot(projected.x - point.x, projected.y - point.y);
}

double Depth(const Pose& pose, const Vec3& xyz) {
  return pose.rotation[6] * xyz.x + pose.rotation[7] * xyz.y + pose.rotation[8] * xyz.z +
         pose.translation.z;
}

}  // namespace

bool EstimateTwoViewGeometry(const py::object& cv2,
                             const FeatureSet& image1,
                             const FeatureSet& image2,
                             const CameraModel& camera1,
                             const CameraModel& camera2,
                             const double max_error,
                             const int min_inliers,
                             std::vector<FeatureMatch>* matches,
                             Mat3* relative_rotation,
                             Vec3* relative_translation) {
  if (matches == nullptr || matches->size() < static_cast<std::size_t>(std::max(5, min_inliers))) {
    return false;
  }
  std::vector<Vec2> points1;
  std::vector<Vec2> points2;
  points1.reserve(matches->size());
  points2.reserve(matches->size());
  for (const FeatureMatch& match : *matches) {
    points1.push_back(image1.points[static_cast<std::size_t>(match.feature1)]);
    points2.push_back(image2.points[static_cast<std::size_t>(match.feature2)]);
  }

  CameraModel shared = camera1;
  shared.focal = 0.5 * (camera1.focal + camera2.focal);
  shared.cx = 0.5 * (camera1.cx + camera2.cx);
  shared.cy = 0.5 * (camera1.cy + camera2.cy);
  const py::array_t<double> matrix = CameraMatrix(shared);
  const py::array_t<double> array1 = PointArray(points1);
  const py::array_t<double> array2 = PointArray(points2);

  const py::tuple essential_result = cv2.attr("findEssentialMat")(
      array1,
      array2,
      matrix,
      py::arg("method") = cv2.attr("RANSAC"),
      py::arg("prob") = 0.999,
      py::arg("threshold") = max_error).cast<py::tuple>();
  if (essential_result.size() < 2 || essential_result[0].is_none()) {
    return false;
  }
  const py::tuple pose_result = cv2.attr("recoverPose")(
      essential_result[0], array1, array2, matrix, py::arg("mask") = essential_result[1]).cast<py::tuple>();
  if (pose_result.size() < 4) {
    return false;
  }
  const auto mask = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>::ensure(pose_result[3]);
  if (!mask) {
    return false;
  }
  std::vector<FeatureMatch> inliers;
  inliers.reserve(matches->size());
  for (py::ssize_t idx = 0; idx < mask.size() && idx < static_cast<py::ssize_t>(matches->size()); ++idx) {
    if (mask.data()[idx] != 0) {
      inliers.push_back((*matches)[static_cast<std::size_t>(idx)]);
    }
  }
  if (inliers.size() < static_cast<std::size_t>(min_inliers)) {
    return false;
  }
  *matches = std::move(inliers);
  *relative_rotation = ReadMat3(pose_result[1]);
  *relative_translation = ReadVec3(pose_result[2]);
  return true;
}

bool EstimateAbsolutePose(const py::object& cv2,
                          const CameraModel& camera,
                          const std::vector<Vec3>& points3d,
                          const std::vector<Vec2>& points2d,
                          const double max_error,
                          const int min_inliers,
                          Pose* pose) {
  if (pose == nullptr || points3d.size() != points2d.size() ||
      points3d.size() < static_cast<std::size_t>(std::max(6, min_inliers))) {
    return false;
  }
  py::array_t<double> object_points(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(points3d.size()), 3});
  auto object = object_points.mutable_unchecked<2>();
  for (py::ssize_t idx = 0; idx < static_cast<py::ssize_t>(points3d.size()); ++idx) {
    const Vec3& point = points3d[static_cast<std::size_t>(idx)];
    object(idx, 0) = point.x;
    object(idx, 1) = point.y;
    object(idx, 2) = point.z;
  }
  const py::array_t<double> image_points = PointArray(points2d);
  py::array_t<double> distortion(std::vector<py::ssize_t>{4, 1});
  std::fill(distortion.mutable_data(), distortion.mutable_data() + 4, 0.0);

  const py::tuple result = cv2.attr("solvePnPRansac")(
      object_points,
      image_points,
      CameraMatrix(camera),
      distortion,
      py::arg("iterationsCount") = 200,
      py::arg("reprojectionError") = max_error,
      py::arg("confidence") = 0.999,
      py::arg("flags") = cv2.attr("SOLVEPNP_EPNP")).cast<py::tuple>();
  if (result.size() < 4 || !result[0].cast<bool>() || result[3].is_none()) {
    return false;
  }
  const auto inliers = py::array::ensure(result[3]);
  if (!inliers || inliers.size() < min_inliers) {
    return false;
  }
  const py::tuple rodrigues = cv2.attr("Rodrigues")(result[1]).cast<py::tuple>();
  pose->rotation = ReadMat3(rodrigues[0]);
  pose->translation = ReadVec3(result[2]);
  pose->registered = true;
  return true;
}

bool Triangulate(const py::object& cv2,
                 const CameraModel& camera1,
                 const CameraModel& camera2,
                 const Pose& pose1,
                 const Pose& pose2,
                 const Vec2& point1,
                 const Vec2& point2,
                 const double max_error,
                 Vec3* xyz,
                 double* mean_error) {
  py::array_t<double> points1(std::vector<py::ssize_t>{2, 1});
  py::array_t<double> points2(std::vector<py::ssize_t>{2, 1});
  points1.mutable_at(0, 0) = point1.x;
  points1.mutable_at(1, 0) = point1.y;
  points2.mutable_at(0, 0) = point2.x;
  points2.mutable_at(1, 0) = point2.y;
  const py::object homogeneous =
      cv2.attr("triangulatePoints")(ProjectionMatrix(camera1, pose1), ProjectionMatrix(camera2, pose2),
                                     points1, points2);
  const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(homogeneous);
  if (!array || array.size() < 4 || std::abs(array.data()[3]) < 1e-12) {
    return false;
  }
  const double inv_w = 1.0 / array.data()[3];
  const Vec3 candidate{array.data()[0] * inv_w, array.data()[1] * inv_w, array.data()[2] * inv_w};
  if (!std::isfinite(candidate.x) || !std::isfinite(candidate.y) || !std::isfinite(candidate.z) ||
      Depth(pose1, candidate) <= 0.0 || Depth(pose2, candidate) <= 0.0) {
    return false;
  }
  const double error1 = ReprojectionError(camera1, pose1, candidate, point1);
  const double error2 = ReprojectionError(camera2, pose2, candidate, point2);
  if (error1 > max_error || error2 > max_error) {
    return false;
  }
  *xyz = candidate;
  *mean_error = 0.5 * (error1 + error2);
  return true;
}

Pose ComposeRelativePose(const Pose& world_to_camera1,
                         const Mat3& camera2_from_camera1,
                         const Vec3& translation) {
  Pose result;
  result.rotation = Multiply(camera2_from_camera1, world_to_camera1.rotation);
  const Vec3 rotated_translation = Multiply(camera2_from_camera1, world_to_camera1.translation);
  result.translation = {rotated_translation.x + translation.x,
                        rotated_translation.y + translation.y,
                        rotated_translation.z + translation.z};
  result.registered = true;
  return result;
}

Vec2 Project(const CameraModel& camera, const Pose& pose, const Vec3& xyz) {
  const Vec3 rotated = Multiply(pose.rotation, xyz);
  const Vec3 camera_point{rotated.x + pose.translation.x,
                          rotated.y + pose.translation.y,
                          rotated.z + pose.translation.z};
  if (std::abs(camera_point.z) < 1e-12) {
    return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  }
  double nx = camera_point.x / camera_point.z;
  double ny = camera_point.y / camera_point.z;
  const double radial_scale = 1.0 + camera.radial * (nx * nx + ny * ny);
  nx *= radial_scale;
  ny *= radial_scale;
  return {camera.focal * nx + camera.cx, camera.focal * ny + camera.cy};
}

Vec3 CameraCenter(const Pose& pose) {
  const Mat3 inverse_rotation = Transpose(pose.rotation);
  const Vec3 inverse_translation = Multiply(inverse_rotation, pose.translation);
  return {-inverse_translation.x, -inverse_translation.y, -inverse_translation.z};
}

double Distance(const Vec3& lhs, const Vec3& rhs) {
  return std::sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y) +
                   (lhs.z - rhs.z) * (lhs.z - rhs.z));
}

std::array<double, 4> RotationToQuaternion(const Mat3& rotation) {
  // Same scalar-first convention as COLMAP's images.txt.
  std::array<double, 4> quaternion{};
  const double trace = rotation[0] + rotation[4] + rotation[8];
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    quaternion = {0.25 * s, (rotation[7] - rotation[5]) / s,
                  (rotation[2] - rotation[6]) / s, (rotation[3] - rotation[1]) / s};
  } else if (rotation[0] > rotation[4] && rotation[0] > rotation[8]) {
    const double s = std::sqrt(1.0 + rotation[0] - rotation[4] - rotation[8]) * 2.0;
    quaternion = {(rotation[7] - rotation[5]) / s, 0.25 * s,
                  (rotation[1] + rotation[3]) / s, (rotation[2] + rotation[6]) / s};
  } else if (rotation[4] > rotation[8]) {
    const double s = std::sqrt(1.0 + rotation[4] - rotation[0] - rotation[8]) * 2.0;
    quaternion = {(rotation[2] - rotation[6]) / s, (rotation[1] + rotation[3]) / s,
                  0.25 * s, (rotation[5] + rotation[7]) / s};
  } else {
    const double s = std::sqrt(1.0 + rotation[8] - rotation[0] - rotation[4]) * 2.0;
    quaternion = {(rotation[3] - rotation[1]) / s, (rotation[2] + rotation[6]) / s,
                  (rotation[5] + rotation[7]) / s, 0.25 * s};
  }
  const double norm = std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  for (double& value : quaternion) {
    value /= norm;
  }
  return quaternion;
}

}  // namespace mdic::sfm
