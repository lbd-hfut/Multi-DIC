find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
  set(MDIC_DEFAULT_COLMAP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/upstream")
  set(MDIC_COLMAP_SOURCE_DIR "" CACHE PATH "Optional COLMAP source tree used to build the embedded CPU SfM backend.")
  set(MDIC_WITH_EMBEDDED_COLMAP OFF)

  if(NOT MDIC_COLMAP_SOURCE_DIR AND EXISTS "${MDIC_DEFAULT_COLMAP_SOURCE_DIR}/CMakeLists.txt")
    set(MDIC_COLMAP_SOURCE_DIR "${MDIC_DEFAULT_COLMAP_SOURCE_DIR}")
  endif()

  if(MDIC_COLMAP_SOURCE_DIR)
    get_filename_component(MDIC_COLMAP_SOURCE_DIR "${MDIC_COLMAP_SOURCE_DIR}" ABSOLUTE)
    if(EXISTS "${MDIC_COLMAP_SOURCE_DIR}/CMakeLists.txt")
      set(CUDA_ENABLED OFF CACHE BOOL "" FORCE)
      set(GUI_ENABLED OFF CACHE BOOL "" FORCE)
      set(MVS_ENABLED OFF CACHE BOOL "" FORCE)
      set(OPENGL_ENABLED OFF CACHE BOOL "" FORCE)
      set(ONNX_ENABLED OFF CACHE BOOL "" FORCE)
      set(CGAL_ENABLED OFF CACHE BOOL "" FORCE)
      set(LSD_ENABLED OFF CACHE BOOL "" FORCE)
      set(TESTS_ENABLED OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLED OFF CACHE BOOL "" FORCE)
      set(UNINSTALL_ENABLED OFF CACHE BOOL "" FORCE)
      set(DOWNLOAD_ENABLED OFF CACHE BOOL "" FORCE)
      set(MDIC_COLMAP_THIRDPARTY_DIR "${MDIC_COLMAP_SOURCE_DIR}/src/thirdparty")
      if(EXISTS "${MDIC_COLMAP_THIRDPARTY_DIR}/PoseLib/CMakeLists.txt")
        set(FETCHCONTENT_SOURCE_DIR_POSELIB "${MDIC_COLMAP_THIRDPARTY_DIR}/PoseLib" CACHE PATH "" FORCE)
      endif()
      if(EXISTS "${MDIC_COLMAP_THIRDPARTY_DIR}/faiss/CMakeLists.txt")
        set(FETCHCONTENT_SOURCE_DIR_FAISS "${MDIC_COLMAP_THIRDPARTY_DIR}/faiss" CACHE PATH "" FORCE)
      endif()
      add_subdirectory("${MDIC_COLMAP_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/colmap-src" EXCLUDE_FROM_ALL)
      set(MDIC_WITH_EMBEDDED_COLMAP ON)
    else()
      message(FATAL_ERROR "MDIC_COLMAP_SOURCE_DIR does not point to a COLMAP source tree: ${MDIC_COLMAP_SOURCE_DIR}")
    endif()
  endif()

  pybind11_add_module(native_colmap src/colmap_bindings.cpp)
  target_compile_features(native_colmap PRIVATE cxx_std_17)
  if(MDIC_WITH_EMBEDDED_COLMAP)
    target_compile_definitions(native_colmap PRIVATE MDIC_WITH_EMBEDDED_COLMAP=1)
    target_include_directories(native_colmap PRIVATE "${MDIC_COLMAP_SOURCE_DIR}/src")
    target_link_libraries(native_colmap
      PRIVATE
        colmap_controllers
        colmap_feature
        colmap_sfm
        colmap_scene
        colmap_sensor
        colmap_util
        colmap_vlfeat
    )
  endif()
  if(MSVC)
    target_compile_options(native_colmap PRIVATE /W4)
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
