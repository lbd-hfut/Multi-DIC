// Stable pybind11 facade for Multi-DIC's compact COLMAP-derived SfM backend.
#include "incremental_sfm.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace {

py::dict RunCpuSfm(const std::string& database_path,
                   const std::string& image_path,
                   const std::string& sparse_path,
                   const std::string& text_path,
                   const std::vector<std::string>& image_names,
                   const py::dict& options) {
  return mdic::sfm::RunCpuSfm(
      database_path, image_path, sparse_path, text_path, image_names, options);
}

}  // namespace

PYBIND11_MODULE(native_colmap, module) {
  module.doc() = "Compact CPU sparse SfM API derived from the COLMAP pipeline for Multi-DIC.";
  module.def("has_embedded_colmap", []() { return true; });
  module.def("capabilities", &mdic::sfm::Capabilities);
  module.def("run_cpu_sfm",
             &RunCpuSfm,
             py::arg("database_path"),
             py::arg("image_path"),
             py::arg("sparse_path"),
             py::arg("text_path"),
             py::arg("image_names"),
             py::arg("options"));
}
