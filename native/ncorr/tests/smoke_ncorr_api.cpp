#include "ncorr_api.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int width = 64;
    constexpr int height = 64;
    multidic::ncorr::GrayImage reference{width, height, std::vector<double>(width * height, 0.0)};
    multidic::ncorr::GrayImage deformed{width, height, std::vector<double>(width * height, 0.0)};
    multidic::ncorr::BoolMask roi{width, height, std::vector<unsigned char>(width * height, 1)};
    auto idx = [](int x, int y) { return static_cast<std::size_t>(y + x * height); };
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            reference.pixels[idx(x, y)] = std::sin(0.17 * x) + std::cos(0.11 * y) + 0.01 * x * y;
        }
    }
    for (int x = 3; x < width; ++x) {
        for (int y = 0; y < height - 2; ++y) {
            deformed.pixels[idx(x, y)] = reference.pixels[idx(x - 3, y + 2)];
        }
    }

    multidic::ncorr::DIC2DConfig config;
    config.subset_radius = 10;
    config.subset_spacing = 0;
    config.subset_truncation = true;
    config.roi_min_region_area = 0;

    const auto seed = multidic::ncorr::select_first_seed_inside_roi(roi, {{30.0, 30.0}}, config);
    const auto result = multidic::ncorr::run_ncorr_dic2d(reference, deformed, roi, seed, config);

    if (seed.xy.x != 30.0 || seed.xy.y != 30.0) {
        std::cerr << "Unexpected seed point: " << seed.xy.x << ", " << seed.xy.y << "\n";
        return 1;
    }
    if (!result.ok) {
        std::cerr << "Native reduced-grid propagation should report success.\n";
        return 1;
    }
    if (result.reference_points.empty()) {
        std::cerr << "Expected nonempty reference grid.\n";
        return 1;
    }
    if (result.roi_region_count != 1 || result.seed_region != 0) {
        std::cerr << "Unexpected region metadata: regions=" << result.roi_region_count
                  << " seed_region=" << result.seed_region << "\n";
        return 1;
    }
    if (result.seed_paramvector.size() != 9 ||
        std::abs(result.seed_paramvector[2] - 3.0) > 0.05 ||
        std::abs(result.seed_paramvector[3] + 2.0) > 0.05 ||
        std::abs(result.seed_paramvector[4]) > 0.02 ||
        std::abs(result.seed_paramvector[5]) > 0.02 ||
        std::abs(result.seed_paramvector[6]) > 0.02 ||
        std::abs(result.seed_paramvector[7]) > 0.02 ||
        result.seed_paramvector[8] < 0.99) {
        std::cerr << "Unexpected NCC seed paramvector: u=" << result.seed_paramvector[2]
                  << " v=" << result.seed_paramvector[3]
                  << " corr=" << result.seed_paramvector[8] << "\n";
        return 1;
    }
    std::size_t valid_count = 0;
    for (const auto valid : result.valid) {
        valid_count += valid != 0 ? 1 : 0;
    }
    if (valid_count == 0 ||
        result.u.size() != result.reference_points.size() ||
        result.v.size() != result.reference_points.size() ||
        result.ux.size() != result.reference_points.size() ||
        result.uy.size() != result.reference_points.size() ||
        result.vx.size() != result.reference_points.size() ||
        result.vy.size() != result.reference_points.size() ||
        result.deformed_points.size() != result.reference_points.size()) {
        std::cerr << "Expected propagated displacement arrays.\n";
        return 1;
    }
    std::cout << "seed=" << seed.xy.x << "," << seed.xy.y << "; valid=" << valid_count << "\n";
    return 0;
}
