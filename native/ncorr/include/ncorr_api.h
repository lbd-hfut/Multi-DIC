#ifndef MULTIDIC_NCORR_API_H
#define MULTIDIC_NCORR_API_H

#include <cstdint>
#include <string>
#include <vector>

namespace multidic::ncorr {

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<double> pixels;
};

struct BoolMask {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> values;
};

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

struct SeedPoint {
    Point2D xy;
    Point2D observation_xy;
};

struct DIC2DConfig {
    int subset_radius = 20;
    int subset_spacing = 5;
    int seed_search_radius = 50;
    int rg_search_radius = 1;
    int roi_min_region_area = 2000;
    double cutoff_diffnorm = 1.0e-6;
    int cutoff_iteration = 50;
    int num_threads = 1;
    bool subset_truncation = false;
    bool step_analysis_enabled = false;
    std::string step_analysis_type = "seed";
    bool step_analysis_auto = true;
    int step_analysis_step = 5;
    double units_per_pixel = 1.0;
    double cutoff_corrcoef = 0.6;
    double lenscoef = 0.0;
};

struct RoiRun {
    int x = 0;
    int y_top = 0;
    int y_bottom = 0;
};

struct RoiRegion {
    int upperbound = 0;
    int lowerbound = 0;
    int leftbound = 0;
    int rightbound = 0;
    int totalpoints = 0;
    std::vector<RoiRun> runs;
};

struct SeedInfo {
    std::vector<double> paramvector;
    int num_region = 0;
    int num_thread = 0;
    int computepoints = 0;
};

struct PreparedNcorrInputs {
    int reduced_width = 0;
    int reduced_height = 0;
    std::vector<RoiRegion> regions;
    std::vector<int> thread_diagram;
    SeedInfo seedinfo;
};

struct NcorrResult {
    bool ok = false;
    std::string message;
    int reduced_width = 0;
    int reduced_height = 0;
    int roi_region_count = 0;
    int seed_region = 0;
    std::vector<double> seed_paramvector;
    std::vector<Point2D> reference_points;
    std::vector<Point2D> deformed_points;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> ux;
    std::vector<double> uy;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> corrcoef;
    std::vector<std::uint8_t> valid;
};

bool point_inside_mask(const BoolMask& mask, const Point2D& point);

Point2D snap_to_ncorr_grid_inside_roi(
    const BoolMask& roi_mask,
    const Point2D& observation,
    const DIC2DConfig& config);

SeedPoint select_first_seed_inside_roi(
    const BoolMask& roi_mask,
    const std::vector<Point2D>& colmap_observations,
    const DIC2DConfig& config = {});

std::vector<Point2D> make_reference_grid_points(
    const BoolMask& roi_mask,
    const DIC2DConfig& config);

std::vector<RoiRegion> form_roi_regions(
    const BoolMask& roi_mask,
    const DIC2DConfig& config);

PreparedNcorrInputs prepare_ncorr_inputs(
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config);

SeedInfo estimate_seed_initial_guess(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config);

void validate_inputs(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config);

NcorrResult run_ncorr_dic2d(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config);

}  // namespace multidic::ncorr

#endif  // MULTIDIC_NCORR_API_H
