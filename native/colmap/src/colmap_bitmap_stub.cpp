// Minimal Bitmap symbols for the sparse-only COLMAP port.
//
// COLMAP's reconstruction object contains optional color extraction helpers
// that reference Bitmap. Multi-DIC's native sparse path never reads image
// colors through COLMAP, so these definitions avoid linking OpenImageIO while
// keeping the sparse Reconstruction object linkable.
#include "colmap/sensor/bitmap.h"

namespace colmap {

Bitmap::Bitmap()
    : width_(0), height_(0), channels_(0), linear_colorspace_(false) {}

std::optional<BitmapColor<float>> Bitmap::InterpolateBilinear(double,
                                                              double) const {
  return std::nullopt;
}

bool Bitmap::Read(const std::filesystem::path&, bool, bool) {
  return false;
}

}  // namespace colmap
