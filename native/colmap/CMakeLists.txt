if(WIN32)
  set(_mdic_colmap_default_deps_dir
      "${CMAKE_CURRENT_SOURCE_DIR}/.deps/Library")
else()
  set(_mdic_colmap_default_deps_dir "")
endif()
set(MDIC_COLMAP_DEPS_DIR "${_mdic_colmap_default_deps_dir}" CACHE PATH
    "Project-local CPU dependency prefix for the COLMAP sparse source port.")
if(MDIC_COLMAP_DEPS_DIR AND EXISTS "${MDIC_COLMAP_DEPS_DIR}")
  list(PREPEND CMAKE_PREFIX_PATH
    "${MDIC_COLMAP_DEPS_DIR}"
    "${MDIC_COLMAP_DEPS_DIR}/lib/cmake"
    "${MDIC_COLMAP_DEPS_DIR}/share/eigen3/cmake")
endif()

find_package(pybind11 CONFIG QUIET)
find_package(Eigen3 CONFIG QUIET)
find_package(Ceres CONFIG QUIET)
find_package(glog CONFIG QUIET)
find_package(Boost CONFIG QUIET COMPONENTS graph)
if(NOT TARGET SuiteSparse::SuiteSparseConfig AND
   NOT TARGET SuiteSparse::SuiteSparseConfig_static)
  find_package(SuiteSparse_config CONFIG QUIET)
endif()
if(NOT TARGET SuiteSparse::AMD AND NOT TARGET SuiteSparse::AMD_static)
  find_package(AMD CONFIG QUIET)
endif()
if(NOT TARGET SuiteSparse::COLAMD AND NOT TARGET SuiteSparse::COLAMD_static)
  find_package(COLAMD CONFIG QUIET)
endif()
if(NOT TARGET SuiteSparse::CAMD AND NOT TARGET SuiteSparse::CAMD_static)
  find_package(CAMD CONFIG QUIET)
endif()
if(NOT TARGET SuiteSparse::CCOLAMD AND NOT TARGET SuiteSparse::CCOLAMD_static)
  find_package(CCOLAMD CONFIG QUIET)
endif()
if(NOT TARGET SuiteSparse::CHOLMOD AND NOT TARGET SuiteSparse::CHOLMOD_static)
  find_package(CHOLMOD CONFIG QUIET)
endif()
find_package(SQLite3 QUIET)
foreach(_suite_component SuiteSparse_config AMD COLAMD CAMD CCOLAMD)
  if(EXISTS "${MDIC_COLMAP_DEPS_DIR}/lib/cmake/${_suite_component}/${_suite_component}Config.cmake")
    include("${MDIC_COLMAP_DEPS_DIR}/lib/cmake/${_suite_component}/${_suite_component}Config.cmake")
  endif()
endforeach()
if(NOT TARGET SuiteSparse::CHOLMOD AND EXISTS "${MDIC_COLMAP_DEPS_DIR}/lib/cmake/CHOLMOD/CHOLMODTargets.cmake")
  include("${MDIC_COLMAP_DEPS_DIR}/lib/cmake/CHOLMOD/CHOLMODTargets.cmake")
endif()

set(MDIC_COLMAP_CHOLMOD_TARGET "")
foreach(_cholmod_target SuiteSparse::CHOLMOD SuiteSparse::CHOLMOD_static)
  if(TARGET "${_cholmod_target}")
    set(MDIC_COLMAP_CHOLMOD_TARGET "${_cholmod_target}")
    break()
  endif()
endforeach()

set(MDIC_COLMAP_SQLITE_TARGET "")
foreach(_sqlite_target SQLite3::SQLite3 SQLite::SQLite3 unofficial::sqlite3::sqlite3)
  if(TARGET "${_sqlite_target}")
    set(MDIC_COLMAP_SQLITE_TARGET "${_sqlite_target}")
    break()
  endif()
endforeach()

set(COLMAP_PORT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(EXISTS "${COLMAP_PORT_SRC_DIR}/thirdparty/PoseLib/CMakeLists.txt" AND NOT TARGET PoseLib::PoseLib)
  set(PYTHON_PACKAGE OFF CACHE BOOL "" FORCE)
  set(WITH_BENCHMARK OFF CACHE BOOL "" FORCE)
  add_subdirectory("${COLMAP_PORT_SRC_DIR}/thirdparty/PoseLib"
                   "${CMAKE_CURRENT_BINARY_DIR}/poselib" EXCLUDE_FROM_ALL)
  set_target_properties(PoseLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TARGET PoseLib AND NOT TARGET PoseLib::PoseLib)
    add_library(PoseLib::PoseLib ALIAS PoseLib)
  endif()
endif()

set(MDIC_COLMAP_SPARSE_READY OFF)
if(Eigen3_FOUND
   AND Ceres_FOUND
   AND TARGET glog::glog
   AND Boost_FOUND
   AND MDIC_COLMAP_CHOLMOD_TARGET
   AND SQLite3_FOUND
   AND MDIC_COLMAP_SQLITE_TARGET
   AND TARGET PoseLib::PoseLib
   AND EXISTS "${COLMAP_PORT_SRC_DIR}/colmap/sfm/incremental_mapper.cc")
  set(MDIC_COLMAP_SPARSE_READY ON)
endif()

if(MDIC_COLMAP_SPARSE_READY)
  add_library(mdic_colmap_sparse STATIC
    # Feature/scene data types and utility support used by the mapper.
    "${COLMAP_PORT_SRC_DIR}/colmap/feature/types.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/colmap_bitmap_stub.cpp"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/base_controller.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/endian.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/file.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/logging.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/misc.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/ply.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/string.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/threading.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/timer.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/util/version.cc"

    # CPU math/geometry/optimization.
    "${COLMAP_PORT_SRC_DIR}/colmap/math/graph_cut.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/math/math.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/math/polynomial.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/math/random.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/math/spanning_tree.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/bbox.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/essential_matrix.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/gps.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/homography_matrix.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/normalization.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/pose.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/pose_prior.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/rigid3.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/sim3.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/geometry/triangulation.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/image/line.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/optim/combination_sampler.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/optim/least_absolute_deviations.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/optim/progressive_sampler.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/optim/random_sampler.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/optim/sprt.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/optim/support_measurement.cc"

    # Camera/rig/database-cache/reconstruction model.
    "${COLMAP_PORT_SRC_DIR}/colmap/sensor/database.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sensor/models.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sensor/rig.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sensor/specs.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/camera.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/correspondence_graph.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/database.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/database_cache.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/database_sqlite.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/frame.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/image.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/point2d.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/point3d.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/pose_graph.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/projection.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_clustering.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_io.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_io_binary.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_io_text.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_io_utils.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_manager.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/reconstruction_pruning.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/rig.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/scene_clustering.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/synthetic.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/track.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/two_view_geometry.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/scene/visibility_pyramid.cc"

    # Estimators, Ceres BA and incremental SfM.
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/alignment.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/bundle_adjustment.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/bundle_adjustment_ceres.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/coordinate_frame.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/covariance.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/generalized_pose.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/global_positioning.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/gravity_refinement.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/pose.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/rotation_averaging.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/rotation_averaging_impl.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/triangulation.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/two_view_geometry.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/view_graph_calibration.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/absolute_pose.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/affine_transform.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/essential_matrix.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/fundamental_matrix.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/generalized_absolute_pose.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/generalized_relative_pose.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/homography_matrix.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/poselib_utils.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/estimators/solvers/similarity_transform.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sfm/incremental_mapper.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sfm/incremental_mapper_impl.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sfm/incremental_triangulator.cc"
    "${COLMAP_PORT_SRC_DIR}/colmap/sfm/observation_manager.cc"
  )
  target_compile_features(mdic_colmap_sparse PUBLIC cxx_std_17)
  set_target_properties(mdic_colmap_sparse PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(mdic_colmap_sparse PUBLIC
    "${COLMAP_PORT_SRC_DIR}"
    "${COLMAP_PORT_SRC_DIR}/thirdparty/PoseLib")
  target_compile_definitions(mdic_colmap_sparse PUBLIC
    GLOG_NO_ABBREVIATED_SEVERITIES
    NOMINMAX
    WIN32_LEAN_AND_MEAN)
  if(glog_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
    target_compile_definitions(mdic_colmap_sparse PUBLIC
      GLOG_VERSION_MAJOR=${CMAKE_MATCH_1}
      GLOG_VERSION_MINOR=${CMAKE_MATCH_2})
  endif()
  target_link_libraries(mdic_colmap_sparse PUBLIC
    Eigen3::Eigen
    Ceres::ceres
    glog::glog
    Boost::graph
    "${MDIC_COLMAP_CHOLMOD_TARGET}"
    "${MDIC_COLMAP_SQLITE_TARGET}"
    PoseLib::PoseLib)
  if(MSVC)
    target_compile_options(mdic_colmap_sparse PRIVATE /bigobj /W3 /permissive-)
  endif()
else()
  message(FATAL_ERROR
    "Native COLMAP sparse SfM dependencies are incomplete. Multi-DIC no longer "
    "provides an alternate SfM backend.")
endif()

if(pybind11_FOUND)
  pybind11_add_module(native_colmap
    src/colmap_bindings.cpp
    src/colmap_sparse_mapper.cpp
    src/feature.cpp
    src/geometry.cpp
    src/incremental_sfm.cpp
    src/model_io.cpp
  )
  target_compile_features(native_colmap PRIVATE cxx_std_17)
  target_include_directories(native_colmap PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_definitions(native_colmap PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)
  target_link_libraries(native_colmap PRIVATE mdic_colmap_sparse)
  target_compile_definitions(native_colmap PRIVATE MDIC_HAS_COLMAP_SPARSE_SOURCE=1)
  if(MSVC)
    target_compile_options(native_colmap PRIVATE /W4 /permissive-)
  else()
    target_compile_options(native_colmap PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  install(TARGETS native_colmap
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
  )
else()
  message(STATUS "pybind11 was not found; native_colmap Python module will not be built.")
endif()
