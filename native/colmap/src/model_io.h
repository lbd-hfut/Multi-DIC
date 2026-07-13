// COLMAP text model writer for the exact files consumed by Multi-DIC.
#pragma once

#include "sfm_types.h"

#include <filesystem>

namespace mdic::sfm {

void WriteTextModel(const Reconstruction& reconstruction,
                    const std::filesystem::path& model_path);

}  // namespace mdic::sfm
