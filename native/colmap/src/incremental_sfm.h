// Stable project-facing API for the compact COLMAP-derived CPU SfM backend.
#pragma once

#include <pybind11/pybind11.h>

#include <string>
#include <vector>

namespace mdic::sfm {

pybind11::dict RunCpuSfm(const std::string& database_path,
                         const std::string& image_path,
                         const std::string& sparse_path,
                         const std::string& text_path,
                         const std::vector<std::string>& image_names,
                         const pybind11::dict& options);

pybind11::dict Capabilities();

}  // namespace mdic::sfm
