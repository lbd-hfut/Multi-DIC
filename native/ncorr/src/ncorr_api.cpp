#include "ncorr_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace multidic::ncorr {
namespace {

bool finite_point(const Point2D& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

void validate_image(const GrayImage& image, const char* name) {
    if (image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument(std::string(name) + " dimensions must be positive.");
    }
    const auto expected = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
    if (image.pixels.size() != expected) {
        throw std::invalid_argument(std::string(name) + " pixel count does not match width*height.");
    }
}

void validate_mask(const BoolMask& mask) {
    if (mask.width <= 0 || mask.height <= 0) {
        throw std::invalid_argument("ROI mask dimensions must be positive.");
    }
    const auto expected = static_cast<std::size_t>(mask.width) * static_cast<std::size_t>(mask.height);
    if (mask.values.size() != expected) {
        throw std::invalid_argument("ROI mask value count does not match width*height.");
    }
}

int grid_step(const DIC2DConfig& config) {
    return config.subset_spacing + 1;
}

int clamp_int(const int value, const int lo, const int hi) {
    return std::max(lo, std::min(value, hi));
}

bool subset_center_is_allowed(const BoolMask& mask, const int x, const int y, const DIC2DConfig& config) {
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) {
        return false;
    }
    const auto idx = static_cast<std::size_t>(y) + static_cast<std::size_t>(x) * static_cast<std::size_t>(mask.height);
    if (mask.values[idx] == 0) {
        return false;
    }
    if (config.subset_truncation) {
        return true;
    }
    return x >= config.subset_radius && y >= config.subset_radius &&
           x < mask.width - config.subset_radius && y < mask.height - config.subset_radius;
}

std::size_t column_major_index(const int height, const int x, const int y) {
    return static_cast<std::size_t>(y) + static_cast<std::size_t>(x) * static_cast<std::size_t>(height);
}

double pixel_at(const GrayImage& image, const int x, const int y) {
    return image.pixels[column_major_index(image.height, x, y)];
}

bool bilinear_at(double& value, const GrayImage& image, const double x, const double y) {
    if (x < 0.0 || y < 0.0 || x >= static_cast<double>(image.width - 1) || y >= static_cast<double>(image.height - 1)) {
        return false;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double ax = x - static_cast<double>(x0);
    const double ay = y - static_cast<double>(y0);
    const double g00 = pixel_at(image, x0, y0);
    const double g10 = pixel_at(image, x0 + 1, y0);
    const double g01 = pixel_at(image, x0, y0 + 1);
    const double g11 = pixel_at(image, x0 + 1, y0 + 1);
    value = (1.0 - ax) * (1.0 - ay) * g00 +
            ax * (1.0 - ay) * g10 +
            (1.0 - ax) * ay * g01 +
            ax * ay * g11;
    return true;
}

bool gradient_at(double& gx, double& gy, const GrayImage& image, const double x, const double y) {
    double right = 0.0;
    double left = 0.0;
    double down = 0.0;
    double up = 0.0;
    if (!bilinear_at(right, image, x + 0.5, y) ||
        !bilinear_at(left, image, x - 0.5, y) ||
        !bilinear_at(down, image, x, y + 0.5) ||
        !bilinear_at(up, image, x, y - 0.5)) {
        return false;
    }
    gx = right - left;
    gy = down - up;
    return true;
}

struct AffineParams {
    double u = 0.0;
    double v = 0.0;
    double ux = 0.0;
    double uy = 0.0;
    double vx = 0.0;
    double vy = 0.0;
};

bool solve_6x6(std::array<double, 6>& x, std::array<double, 36> matrix, std::array<double, 6> rhs) {
    for (int col = 0; col < 6; ++col) {
        int pivot = col;
        double pivot_abs = std::abs(matrix[static_cast<std::size_t>(col) * 6 + col]);
        for (int row = col + 1; row < 6; ++row) {
            const double candidate = std::abs(matrix[static_cast<std::size_t>(row) * 6 + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= 1.0e-12) {
            return false;
        }
        if (pivot != col) {
            for (int k = col; k < 6; ++k) {
                std::swap(matrix[static_cast<std::size_t>(col) * 6 + k], matrix[static_cast<std::size_t>(pivot) * 6 + k]);
            }
            std::swap(rhs[static_cast<std::size_t>(col)], rhs[static_cast<std::size_t>(pivot)]);
        }
        const double diag = matrix[static_cast<std::size_t>(col) * 6 + col];
        for (int row = col + 1; row < 6; ++row) {
            const double factor = matrix[static_cast<std::size_t>(row) * 6 + col] / diag;
            if (factor == 0.0) {
                continue;
            }
            matrix[static_cast<std::size_t>(row) * 6 + col] = 0.0;
            for (int k = col + 1; k < 6; ++k) {
                matrix[static_cast<std::size_t>(row) * 6 + k] -= factor * matrix[static_cast<std::size_t>(col) * 6 + k];
            }
            rhs[static_cast<std::size_t>(row)] -= factor * rhs[static_cast<std::size_t>(col)];
        }
    }

    for (int row = 5; row >= 0; --row) {
        double value = rhs[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < 6; ++col) {
            value -= matrix[static_cast<std::size_t>(row) * 6 + col] * x[static_cast<std::size_t>(col)];
        }
        const double diag = matrix[static_cast<std::size_t>(row) * 6 + row];
        if (std::abs(diag) <= 1.0e-12) {
            return false;
        }
        x[static_cast<std::size_t>(row)] = value / diag;
    }
    return true;
}

std::vector<std::pair<int, int>> subset_offsets(
    const BoolMask& roi_mask,
    const int center_x,
    const int center_y,
    const DIC2DConfig& config) {
    std::vector<std::pair<int, int>> offsets;
    offsets.reserve(static_cast<std::size_t>((2 * config.subset_radius + 1) * (2 * config.subset_radius + 1)));
    for (int dx = -config.subset_radius; dx <= config.subset_radius; ++dx) {
        for (int dy = -config.subset_radius; dy <= config.subset_radius; ++dy) {
            if (dx * dx + dy * dy > config.subset_radius * config.subset_radius) {
                continue;
            }
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (x < 0 || y < 0 || x >= roi_mask.width || y >= roi_mask.height) {
                continue;
            }
            if (config.subset_truncation && roi_mask.values[column_major_index(roi_mask.height, x, y)] == 0) {
                continue;
            }
            offsets.push_back({dx, dy});
        }
    }
    return offsets;
}

bool ncc_at(
    double& corr,
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const int ref_x,
    const int ref_y,
    const int def_x,
    const int def_y,
    const std::vector<std::pair<int, int>>& offsets) {
    if (offsets.size() < 3) {
        return false;
    }

    double f_mean = 0.0;
    double g_mean = 0.0;
    for (const auto& [dx, dy] : offsets) {
        const int gx = def_x + dx;
        const int gy = def_y + dy;
        if (gx < 0 || gy < 0 || gx >= deformed_image.width || gy >= deformed_image.height) {
            return false;
        }
        f_mean += pixel_at(reference_image, ref_x + dx, ref_y + dy);
        g_mean += pixel_at(deformed_image, gx, gy);
    }
    f_mean /= static_cast<double>(offsets.size());
    g_mean /= static_cast<double>(offsets.size());

    double numerator = 0.0;
    double f_ss = 0.0;
    double g_ss = 0.0;
    for (const auto& [dx, dy] : offsets) {
        const double f_value = pixel_at(reference_image, ref_x + dx, ref_y + dy) - f_mean;
        const double g_value = pixel_at(deformed_image, def_x + dx, def_y + dy) - g_mean;
        numerator += f_value * g_value;
        f_ss += f_value * f_value;
        g_ss += g_value * g_value;
    }
    if (f_ss <= std::numeric_limits<double>::epsilon() || g_ss <= std::numeric_limits<double>::epsilon()) {
        return false;
    }
    corr = numerator / std::sqrt(f_ss * g_ss);
    return std::isfinite(corr);
}

double zncc_for_translation(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const int ref_x,
    const int ref_y,
    const double u,
    const double v,
    const std::vector<std::pair<int, int>>& offsets) {
    std::vector<double> f_values;
    std::vector<double> g_values;
    f_values.reserve(offsets.size());
    g_values.reserve(offsets.size());
    double f_mean = 0.0;
    double g_mean = 0.0;
    for (const auto& [dx, dy] : offsets) {
        double g_value = 0.0;
        if (!bilinear_at(g_value, deformed_image, static_cast<double>(ref_x + dx) + u, static_cast<double>(ref_y + dy) + v)) {
            continue;
        }
        const double f_value = pixel_at(reference_image, ref_x + dx, ref_y + dy);
        f_values.push_back(f_value);
        g_values.push_back(g_value);
        f_mean += f_value;
        g_mean += g_value;
    }
    if (f_values.size() < 3) {
        return -std::numeric_limits<double>::infinity();
    }
    f_mean /= static_cast<double>(f_values.size());
    g_mean /= static_cast<double>(g_values.size());

    double numerator = 0.0;
    double f_ss = 0.0;
    double g_ss = 0.0;
    for (std::size_t i = 0; i < f_values.size(); ++i) {
        const double f = f_values[i] - f_mean;
        const double g = g_values[i] - g_mean;
        numerator += f * g;
        f_ss += f * f;
        g_ss += g * g;
    }
    if (f_ss <= std::numeric_limits<double>::epsilon() || g_ss <= std::numeric_limits<double>::epsilon()) {
        return -std::numeric_limits<double>::infinity();
    }
    return numerator / std::sqrt(f_ss * g_ss);
}

double zncc_for_affine(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const int ref_x,
    const int ref_y,
    const AffineParams& params,
    const std::vector<std::pair<int, int>>& offsets) {
    std::vector<double> f_values;
    std::vector<double> g_values;
    f_values.reserve(offsets.size());
    g_values.reserve(offsets.size());
    double f_mean = 0.0;
    double g_mean = 0.0;
    for (const auto& [dx_int, dy_int] : offsets) {
        const double dx = static_cast<double>(dx_int);
        const double dy = static_cast<double>(dy_int);
        double g_value = 0.0;
        if (!bilinear_at(
                g_value,
                deformed_image,
                static_cast<double>(ref_x + dx_int) + params.u + params.ux * dx + params.uy * dy,
                static_cast<double>(ref_y + dy_int) + params.v + params.vx * dx + params.vy * dy)) {
            continue;
        }
        const double f_value = pixel_at(reference_image, ref_x + dx_int, ref_y + dy_int);
        f_values.push_back(f_value);
        g_values.push_back(g_value);
        f_mean += f_value;
        g_mean += g_value;
    }
    if (f_values.size() < 6) {
        return -std::numeric_limits<double>::infinity();
    }
    f_mean /= static_cast<double>(f_values.size());
    g_mean /= static_cast<double>(g_values.size());

    double numerator = 0.0;
    double f_ss = 0.0;
    double g_ss = 0.0;
    for (std::size_t i = 0; i < f_values.size(); ++i) {
        const double f = f_values[i] - f_mean;
        const double g = g_values[i] - g_mean;
        numerator += f * g;
        f_ss += f * f;
        g_ss += g * g;
    }
    if (f_ss <= std::numeric_limits<double>::epsilon() || g_ss <= std::numeric_limits<double>::epsilon()) {
        return -std::numeric_limits<double>::infinity();
    }
    return numerator / std::sqrt(f_ss * g_ss);
}

bool refine_translation_icgn(
    double& refined_u,
    double& refined_v,
    double& refined_corr,
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const int ref_x,
    const int ref_y,
    const double init_u,
    const double init_v,
    const DIC2DConfig& config) {
    const auto offsets = subset_offsets(roi_mask, ref_x, ref_y, config);
    if (offsets.size() < 3) {
        return false;
    }

    double u = init_u;
    double v = init_v;
    for (int iter = 0; iter < config.cutoff_iteration; ++iter) {
        std::array<double, 4> hessian{0.0, 0.0, 0.0, 0.0};
        std::array<double, 2> gradient{0.0, 0.0};
        int used = 0;
        for (const auto& [dx, dy] : offsets) {
            const double x = static_cast<double>(ref_x + dx) + u;
            const double y = static_cast<double>(ref_y + dy) + v;
            double g_value = 0.0;
            double gx = 0.0;
            double gy = 0.0;
            if (!bilinear_at(g_value, deformed_image, x, y) || !gradient_at(gx, gy, deformed_image, x, y)) {
                continue;
            }
            const double residual = g_value - pixel_at(reference_image, ref_x + dx, ref_y + dy);
            hessian[0] += gx * gx;
            hessian[1] += gx * gy;
            hessian[3] += gy * gy;
            gradient[0] += gx * residual;
            gradient[1] += gy * residual;
            ++used;
        }
        if (used < 3) {
            return false;
        }
        hessian[2] = hessian[1];
        const double det = hessian[0] * hessian[3] - hessian[1] * hessian[2];
        if (std::abs(det) <= 1.0e-12) {
            return false;
        }
        const double delta_u = -(hessian[3] * gradient[0] - hessian[1] * gradient[1]) / det;
        const double delta_v = -(-hessian[2] * gradient[0] + hessian[0] * gradient[1]) / det;
        if (!std::isfinite(delta_u) || !std::isfinite(delta_v)) {
            return false;
        }
        u += delta_u;
        v += delta_v;
        const double diffnorm = std::sqrt(delta_u * delta_u + delta_v * delta_v);
        if (diffnorm < config.cutoff_diffnorm) {
            break;
        }
    }

    const double corr = zncc_for_translation(reference_image, deformed_image, ref_x, ref_y, u, v, offsets);
    if (!std::isfinite(corr)) {
        return false;
    }
    refined_u = u;
    refined_v = v;
    refined_corr = corr;
    return true;
}

bool refine_affine_icgn(
    AffineParams& refined,
    double& refined_corr,
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const int ref_x,
    const int ref_y,
    const AffineParams& init,
    const DIC2DConfig& config) {
    const auto offsets = subset_offsets(roi_mask, ref_x, ref_y, config);
    if (offsets.size() < 6) {
        return false;
    }

    AffineParams params = init;
    for (int iter = 0; iter < config.cutoff_iteration; ++iter) {
        std::array<double, 36> hessian{};
        std::array<double, 6> gradient{};
        int used = 0;
        for (const auto& [dx_int, dy_int] : offsets) {
            const double dx = static_cast<double>(dx_int);
            const double dy = static_cast<double>(dy_int);
            const double x = static_cast<double>(ref_x + dx_int) + params.u + params.ux * dx + params.uy * dy;
            const double y = static_cast<double>(ref_y + dy_int) + params.v + params.vx * dx + params.vy * dy;
            double g_value = 0.0;
            double gx = 0.0;
            double gy = 0.0;
            if (!bilinear_at(g_value, deformed_image, x, y) || !gradient_at(gx, gy, deformed_image, x, y)) {
                continue;
            }
            const double residual = g_value - pixel_at(reference_image, ref_x + dx_int, ref_y + dy_int);
            const std::array<double, 6> jacobian{gx, gy, gx * dx, gx * dy, gy * dx, gy * dy};
            for (int r = 0; r < 6; ++r) {
                gradient[static_cast<std::size_t>(r)] += jacobian[static_cast<std::size_t>(r)] * residual;
                for (int c = 0; c < 6; ++c) {
                    hessian[static_cast<std::size_t>(r) * 6 + c] +=
                        jacobian[static_cast<std::size_t>(r)] * jacobian[static_cast<std::size_t>(c)];
                }
            }
            ++used;
        }
        if (used < 6) {
            return false;
        }

        std::array<double, 6> delta{};
        std::array<double, 6> rhs{};
        for (int i = 0; i < 6; ++i) {
            rhs[static_cast<std::size_t>(i)] = -gradient[static_cast<std::size_t>(i)];
        }
        if (!solve_6x6(delta, hessian, rhs)) {
            return false;
        }
        params.u += delta[0];
        params.v += delta[1];
        params.ux += delta[2];
        params.uy += delta[3];
        params.vx += delta[4];
        params.vy += delta[5];
        double diffnorm = 0.0;
        for (const double value : delta) {
            if (!std::isfinite(value)) {
                return false;
            }
            diffnorm += value * value;
        }
        if (std::sqrt(diffnorm) < config.cutoff_diffnorm) {
            break;
        }
    }

    const double corr = zncc_for_affine(reference_image, deformed_image, ref_x, ref_y, params, offsets);
    if (!std::isfinite(corr)) {
        return false;
    }
    refined = params;
    refined_corr = corr;
    return true;
}

bool estimate_translation_near(
    double& best_u,
    double& best_v,
    double& best_corr,
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const int ref_x,
    const int ref_y,
    const int init_u,
    const int init_v,
    const int search_radius,
    const DIC2DConfig& config) {
    const auto offsets = subset_offsets(roi_mask, ref_x, ref_y, config);
    if (offsets.size() < 3) {
        return false;
    }

    bool found = false;
    best_corr = -std::numeric_limits<double>::infinity();
    best_u = static_cast<double>(init_u);
    best_v = static_cast<double>(init_v);
    for (int du = init_u - search_radius; du <= init_u + search_radius; ++du) {
        for (int dv = init_v - search_radius; dv <= init_v + search_radius; ++dv) {
            double corr = 0.0;
            if (!ncc_at(corr, reference_image, deformed_image, ref_x, ref_y, ref_x + du, ref_y + dv, offsets)) {
                continue;
            }
            if (!found || corr > best_corr) {
                best_corr = corr;
                best_u = static_cast<double>(du);
                best_v = static_cast<double>(dv);
                found = true;
            }
        }
    }
    if (found) {
        double refined_u = best_u;
        double refined_v = best_v;
        double refined_corr = best_corr;
        if (refine_translation_icgn(
                refined_u,
                refined_v,
                refined_corr,
                reference_image,
                deformed_image,
                roi_mask,
                ref_x,
                ref_y,
                best_u,
                best_v,
                config)) {
            best_u = refined_u;
            best_v = refined_v;
            best_corr = refined_corr;
        }
    }
    return found;
}

bool estimate_affine_near(
    AffineParams& best_params,
    double& best_corr,
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const int ref_x,
    const int ref_y,
    const AffineParams& init,
    const int search_radius,
    const DIC2DConfig& config) {
    double best_u = 0.0;
    double best_v = 0.0;
    double corr = 0.0;
    const bool found = estimate_translation_near(
        best_u,
        best_v,
        corr,
        reference_image,
        deformed_image,
        roi_mask,
        ref_x,
        ref_y,
        static_cast<int>(std::lround(init.u)),
        static_cast<int>(std::lround(init.v)),
        search_radius,
        config);
    if (!found) {
        return false;
    }
    AffineParams affine_init = init;
    affine_init.u = best_u;
    affine_init.v = best_v;
    affine_init.ux = init.ux;
    affine_init.uy = init.uy;
    affine_init.vx = init.vx;
    affine_init.vy = init.vy;
    best_params = affine_init;
    best_corr = corr;
    AffineParams refined = affine_init;
    double refined_corr = corr;
    if (refine_affine_icgn(refined, refined_corr, reference_image, deformed_image, roi_mask, ref_x, ref_y, affine_init, config)) {
        best_params = refined;
        best_corr = refined_corr;
    }
    return true;
}

}  // namespace

bool point_inside_mask(const BoolMask& mask, const Point2D& point) {
    if (!finite_point(point)) {
        return false;
    }
    const int x = static_cast<int>(std::lround(point.x));
    const int y = static_cast<int>(std::lround(point.y));
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) {
        return false;
    }
    const auto idx = static_cast<std::size_t>(y) + static_cast<std::size_t>(x) * static_cast<std::size_t>(mask.height);
    return mask.values[idx] != 0;
}

Point2D snap_to_ncorr_grid_inside_roi(
    const BoolMask& roi_mask,
    const Point2D& observation,
    const DIC2DConfig& config) {
    validate_mask(roi_mask);
    if (!finite_point(observation)) {
        throw std::invalid_argument("Observation point must be finite.");
    }
    const int step = grid_step(config);
    if (step <= 0) {
        throw std::invalid_argument("subset_spacing must be non-negative.");
    }

    const int x0 = clamp_int(static_cast<int>(std::lround(observation.x / step)) * step, 0, roi_mask.width - 1);
    const int y0 = clamp_int(static_cast<int>(std::lround(observation.y / step)) * step, 0, roi_mask.height - 1);
    if (subset_center_is_allowed(roi_mask, x0, y0, config)) {
        return Point2D{static_cast<double>(x0), static_cast<double>(y0)};
    }

    const int max_radius = std::max(roi_mask.width, roi_mask.height) / step + 2;
    for (int radius = 1; radius <= max_radius; ++radius) {
        Point2D best;
        double best_dist2 = 0.0;
        bool found = false;
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                if (std::max(std::abs(dx), std::abs(dy)) != radius) {
                    continue;
                }
                const int x = x0 + dx * step;
                const int y = y0 + dy * step;
                if (!subset_center_is_allowed(roi_mask, x, y, config)) {
                    continue;
                }
                const double dist2 = (observation.x - x) * (observation.x - x) +
                                     (observation.y - y) * (observation.y - y);
                if (!found || dist2 < best_dist2) {
                    best = Point2D{static_cast<double>(x), static_cast<double>(y)};
                    best_dist2 = dist2;
                    found = true;
                }
            }
        }
        if (found) {
            return best;
        }
    }
    throw std::runtime_error("No Ncorr grid point near the observation falls inside the ROI mask.");
}

SeedPoint select_first_seed_inside_roi(
    const BoolMask& roi_mask,
    const std::vector<Point2D>& colmap_observations,
    const DIC2DConfig& config) {
    validate_mask(roi_mask);
    for (const Point2D& point : colmap_observations) {
        if (point_inside_mask(roi_mask, point)) {
            return SeedPoint{snap_to_ncorr_grid_inside_roi(roi_mask, point, config), point};
        }
    }
    throw std::runtime_error("No COLMAP observation falls inside the ROI mask.");
}

std::vector<Point2D> make_reference_grid_points(
    const BoolMask& roi_mask,
    const DIC2DConfig& config) {
    validate_mask(roi_mask);
    const int step = grid_step(config);
    if (step <= 0) {
        throw std::invalid_argument("subset_spacing must be non-negative.");
    }

    std::vector<Point2D> points;
    points.reserve(static_cast<std::size_t>(roi_mask.width / step + 1) * static_cast<std::size_t>(roi_mask.height / step + 1));
    for (int x = 0; x < roi_mask.width; x += step) {
        for (int y = 0; y < roi_mask.height; y += step) {
            if (subset_center_is_allowed(roi_mask, x, y, config)) {
                points.push_back(Point2D{static_cast<double>(x), static_cast<double>(y)});
            }
        }
    }
    return points;
}

std::vector<RoiRegion> form_roi_regions(
    const BoolMask& roi_mask,
    const DIC2DConfig& config) {
    validate_mask(roi_mask);
    const int width = roi_mask.width;
    const int height = roi_mask.height;
    std::vector<std::uint8_t> visited(roi_mask.values.size(), 0);
    std::vector<RoiRegion> regions;
    std::queue<std::pair<int, int>> queue;

    for (int x0 = 0; x0 < width; ++x0) {
        for (int y0 = 0; y0 < height; ++y0) {
            const std::size_t start_idx = column_major_index(height, x0, y0);
            if (visited[start_idx] != 0 || roi_mask.values[start_idx] == 0) {
                continue;
            }

            std::vector<std::pair<int, int>> pixels;
            visited[start_idx] = 1;
            queue.push({x0, y0});
            while (!queue.empty()) {
                const auto [x, y] = queue.front();
                queue.pop();
                pixels.push_back({x, y});

                const int nx[4] = {x - 1, x + 1, x, x};
                const int ny[4] = {y, y, y - 1, y + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || ny[k] < 0 || nx[k] >= width || ny[k] >= height) {
                        continue;
                    }
                    const std::size_t idx = column_major_index(height, nx[k], ny[k]);
                    if (visited[idx] == 0 && roi_mask.values[idx] != 0) {
                        visited[idx] = 1;
                        queue.push({nx[k], ny[k]});
                    }
                }
            }

            if (static_cast<int>(pixels.size()) <= config.roi_min_region_area) {
                continue;
            }

            RoiRegion region;
            region.totalpoints = static_cast<int>(pixels.size());
            region.leftbound = width - 1;
            region.rightbound = 0;
            region.upperbound = height - 1;
            region.lowerbound = 0;
            std::vector<std::vector<int>> rows_by_x(static_cast<std::size_t>(width));
            for (const auto& [x, y] : pixels) {
                rows_by_x[static_cast<std::size_t>(x)].push_back(y);
                region.leftbound = std::min(region.leftbound, x);
                region.rightbound = std::max(region.rightbound, x);
                region.upperbound = std::min(region.upperbound, y);
                region.lowerbound = std::max(region.lowerbound, y);
            }

            for (int x = region.leftbound; x <= region.rightbound; ++x) {
                auto& rows = rows_by_x[static_cast<std::size_t>(x)];
                if (rows.empty()) {
                    continue;
                }
                std::sort(rows.begin(), rows.end());
                int y_top = rows.front();
                int y_prev = rows.front();
                for (std::size_t i = 1; i < rows.size(); ++i) {
                    if (rows[i] == y_prev + 1) {
                        y_prev = rows[i];
                    } else {
                        region.runs.push_back(RoiRun{x, y_top, y_prev});
                        y_top = rows[i];
                        y_prev = rows[i];
                    }
                }
                region.runs.push_back(RoiRun{x, y_top, y_prev});
            }
            regions.push_back(std::move(region));
        }
    }

    return regions;
}

PreparedNcorrInputs prepare_ncorr_inputs(
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config) {
    validate_mask(roi_mask);
    const int step = grid_step(config);
    if (step <= 0) {
        throw std::invalid_argument("subset_spacing must be non-negative.");
    }

    PreparedNcorrInputs prepared;
    prepared.reduced_width = static_cast<int>(std::ceil(static_cast<double>(roi_mask.width) / step));
    prepared.reduced_height = static_cast<int>(std::ceil(static_cast<double>(roi_mask.height) / step));
    prepared.regions = form_roi_regions(roi_mask, config);
    if (prepared.regions.empty()) {
        throw std::runtime_error("ROI mask does not contain a region larger than roi_min_region_area.");
    }

    const int seed_x = static_cast<int>(std::lround(seed_point.xy.x));
    const int seed_y = static_cast<int>(std::lround(seed_point.xy.y));
    int seed_region = -1;
    for (std::size_t idx_region = 0; idx_region < prepared.regions.size(); ++idx_region) {
        const RoiRegion& region = prepared.regions[idx_region];
        if (seed_x < region.leftbound || seed_x > region.rightbound || seed_y < region.upperbound || seed_y > region.lowerbound) {
            continue;
        }
        for (const RoiRun& run : region.runs) {
            if (run.x == seed_x && seed_y >= run.y_top && seed_y <= run.y_bottom) {
                seed_region = static_cast<int>(idx_region);
                break;
            }
        }
        if (seed_region >= 0) {
            break;
        }
    }
    if (seed_region < 0) {
        throw std::runtime_error("Seed point is not inside any retained ROI region.");
    }

    prepared.thread_diagram.assign(
        static_cast<std::size_t>(prepared.reduced_width) * static_cast<std::size_t>(prepared.reduced_height),
        -1);
    int computepoints = 0;
    for (int x = 0; x < roi_mask.width; x += step) {
        for (int y = 0; y < roi_mask.height; y += step) {
            if (!subset_center_is_allowed(roi_mask, x, y, config)) {
                continue;
            }
            const int xr = x / step;
            const int yr = y / step;
            prepared.thread_diagram[column_major_index(prepared.reduced_height, xr, yr)] = 0;
            ++computepoints;
        }
    }

    prepared.seedinfo.paramvector.assign(9, 0.0);
    prepared.seedinfo.paramvector[0] = static_cast<double>(seed_x);
    prepared.seedinfo.paramvector[1] = static_cast<double>(seed_y);
    prepared.seedinfo.paramvector[8] = 0.0;
    prepared.seedinfo.num_region = seed_region;
    prepared.seedinfo.num_thread = 0;
    prepared.seedinfo.computepoints = computepoints;
    return prepared;
}

SeedInfo estimate_seed_initial_guess(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config) {
    validate_inputs(reference_image, deformed_image, roi_mask, seed_point, config);
    PreparedNcorrInputs prepared = prepare_ncorr_inputs(roi_mask, seed_point, config);

    const int seed_x = static_cast<int>(std::lround(seed_point.xy.x));
    const int seed_y = static_cast<int>(std::lround(seed_point.xy.y));
    const auto offsets = subset_offsets(roi_mask, seed_x, seed_y, config);
    if (offsets.size() < 3) {
        throw std::runtime_error("Seed subset has too few pixels for NCC.");
    }

    double f_mean = 0.0;
    for (const auto& [dx, dy] : offsets) {
        f_mean += pixel_at(reference_image, seed_x + dx, seed_y + dy);
    }
    f_mean /= static_cast<double>(offsets.size());

    double f_ss = 0.0;
    for (const auto& [dx, dy] : offsets) {
        const double value = pixel_at(reference_image, seed_x + dx, seed_y + dy) - f_mean;
        f_ss += value * value;
    }
    if (f_ss <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Seed subset has near-zero reference texture.");
    }

    double best_corr = -std::numeric_limits<double>::infinity();
    int best_x = seed_x;
    int best_y = seed_y;
    const int search_x0 = config.seed_search_radius > 0
                              ? std::max(config.subset_radius, seed_x - config.seed_search_radius)
                              : config.subset_radius;
    const int search_x1 = config.seed_search_radius > 0
                              ? std::min(deformed_image.width - config.subset_radius, seed_x + config.seed_search_radius + 1)
                              : deformed_image.width - config.subset_radius;
    const int search_y0 = config.seed_search_radius > 0
                              ? std::max(config.subset_radius, seed_y - config.seed_search_radius)
                              : config.subset_radius;
    const int search_y1 = config.seed_search_radius > 0
                              ? std::min(deformed_image.height - config.subset_radius, seed_y + config.seed_search_radius + 1)
                              : deformed_image.height - config.subset_radius;

    for (int candidate_x = search_x0; candidate_x < search_x1; ++candidate_x) {
        for (int candidate_y = search_y0; candidate_y < search_y1; ++candidate_y) {
            double corr = 0.0;
            if (ncc_at(corr, reference_image, deformed_image, seed_x, seed_y, candidate_x, candidate_y, offsets) &&
                corr > best_corr) {
                best_corr = corr;
                best_x = candidate_x;
                best_y = candidate_y;
            }
        }
    }
    if (!std::isfinite(best_corr)) {
        throw std::runtime_error("NCC initial guess failed to find a valid candidate.");
    }

    double seed_u = static_cast<double>(best_x - seed_x);
    double seed_v = static_cast<double>(best_y - seed_y);
    double seed_corr = best_corr;
    double refined_u = seed_u;
    double refined_v = seed_v;
    double refined_corr = seed_corr;
    if (refine_translation_icgn(
            refined_u,
            refined_v,
            refined_corr,
            reference_image,
            deformed_image,
            roi_mask,
            seed_x,
            seed_y,
            seed_u,
            seed_v,
            config)) {
        seed_u = refined_u;
        seed_v = refined_v;
        seed_corr = refined_corr;
    }
    AffineParams seed_affine{seed_u, seed_v, 0.0, 0.0, 0.0, 0.0};
    AffineParams refined_affine = seed_affine;
    double affine_corr = seed_corr;
    if (refine_affine_icgn(
            refined_affine,
            affine_corr,
            reference_image,
            deformed_image,
            roi_mask,
            seed_x,
            seed_y,
            seed_affine,
            config)) {
        seed_affine = refined_affine;
        seed_corr = affine_corr;
    }

    prepared.seedinfo.paramvector[2] = seed_affine.u;
    prepared.seedinfo.paramvector[3] = seed_affine.v;
    prepared.seedinfo.paramvector[4] = seed_affine.ux;
    prepared.seedinfo.paramvector[5] = seed_affine.uy;
    prepared.seedinfo.paramvector[6] = seed_affine.vx;
    prepared.seedinfo.paramvector[7] = seed_affine.vy;
    prepared.seedinfo.paramvector[8] = seed_corr;
    return prepared.seedinfo;
}

void validate_inputs(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config) {
    validate_image(reference_image, "Reference image");
    validate_image(deformed_image, "Deformed image");
    validate_mask(roi_mask);
    if (reference_image.width != deformed_image.width || reference_image.height != deformed_image.height) {
        throw std::invalid_argument("Reference and deformed images must have the same dimensions.");
    }
    if (roi_mask.width != reference_image.width || roi_mask.height != reference_image.height) {
        throw std::invalid_argument("ROI mask dimensions must match the reference image.");
    }
    if (config.subset_radius < 10) {
        throw std::invalid_argument("subset_radius must be at least 10 to match Ncorr parameter bounds.");
    }
    if (config.subset_spacing < 0) {
        throw std::invalid_argument("subset_spacing must be non-negative.");
    }
    if (config.seed_search_radius < 0) {
        throw std::invalid_argument("seed_search_radius must be non-negative.");
    }
    if (config.rg_search_radius < 0) {
        throw std::invalid_argument("rg_search_radius must be non-negative.");
    }
    if (config.roi_min_region_area < 0) {
        throw std::invalid_argument("roi_min_region_area must be non-negative.");
    }
    if (!point_inside_mask(roi_mask, seed_point.xy)) {
        throw std::invalid_argument("Seed point must be finite, in bounds, and inside the ROI mask.");
    }
    const int step = grid_step(config);
    if (std::fmod(seed_point.xy.x, static_cast<double>(step)) != 0.0 ||
        std::fmod(seed_point.xy.y, static_cast<double>(step)) != 0.0) {
        throw std::invalid_argument("Seed point must be aligned to the Ncorr reduced grid.");
    }
    if (config.cutoff_diffnorm <= 0.0 || config.cutoff_diffnorm > 1.0e-2) {
        throw std::invalid_argument("cutoff_diffnorm must be in (0, 1e-2].");
    }
    if (config.cutoff_iteration < 5) {
        throw std::invalid_argument("cutoff_iteration must be at least 5.");
    }
    if (config.num_threads < 1) {
        throw std::invalid_argument("num_threads must be positive.");
    }
    if (config.units_per_pixel <= 0.0) {
        throw std::invalid_argument("units_per_pixel must be positive.");
    }
}

NcorrResult run_ncorr_dic2d(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config) {
    validate_inputs(reference_image, deformed_image, roi_mask, seed_point, config);

    NcorrResult result;
    result.ok = false;
    const PreparedNcorrInputs prepared = prepare_ncorr_inputs(roi_mask, seed_point, config);
    result.reduced_width = prepared.reduced_width;
    result.reduced_height = prepared.reduced_height;
    result.roi_region_count = static_cast<int>(prepared.regions.size());
    result.seed_region = prepared.seedinfo.num_region;
    result.seed_paramvector = estimate_seed_initial_guess(reference_image, deformed_image, roi_mask, seed_point, config).paramvector;
    result.reference_points = make_reference_grid_points(roi_mask, config);
    result.deformed_points.assign(result.reference_points.size(), {});
    result.u.assign(result.reference_points.size(), 0.0);
    result.v.assign(result.reference_points.size(), 0.0);
    result.ux.assign(result.reference_points.size(), 0.0);
    result.uy.assign(result.reference_points.size(), 0.0);
    result.vx.assign(result.reference_points.size(), 0.0);
    result.vy.assign(result.reference_points.size(), 0.0);
    result.corrcoef.assign(result.reference_points.size(), 0.0);
    result.valid.assign(result.reference_points.size(), 0);

    const int step = grid_step(config);
    std::vector<int> point_index(
        static_cast<std::size_t>(prepared.reduced_width) * static_cast<std::size_t>(prepared.reduced_height),
        -1);
    for (std::size_t i = 0; i < result.reference_points.size(); ++i) {
        const int xr = static_cast<int>(std::lround(result.reference_points[i].x)) / step;
        const int yr = static_cast<int>(std::lround(result.reference_points[i].y)) / step;
        point_index[column_major_index(prepared.reduced_height, xr, yr)] = static_cast<int>(i);
    }

    struct QueueItem {
        int point = -1;
        AffineParams params;
        double priority = 0.0;
    };
    struct QueueLess {
        bool operator()(const QueueItem& a, const QueueItem& b) const {
            return a.priority < b.priority;
        }
    };

    std::priority_queue<QueueItem, std::vector<QueueItem>, QueueLess> queue;
    const int seed_x = static_cast<int>(std::lround(seed_point.xy.x));
    const int seed_y = static_cast<int>(std::lround(seed_point.xy.y));
    const int seed_xr = seed_x / step;
    const int seed_yr = seed_y / step;
    const int seed_idx = point_index[column_major_index(prepared.reduced_height, seed_xr, seed_yr)];
    if (seed_idx < 0) {
        throw std::runtime_error("Seed point is not part of the reduced reference grid.");
    }
    result.u[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[2];
    result.v[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[3];
    result.ux[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[4];
    result.uy[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[5];
    result.vx[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[6];
    result.vy[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[7];
    result.corrcoef[static_cast<std::size_t>(seed_idx)] = result.seed_paramvector[8];
    result.deformed_points[static_cast<std::size_t>(seed_idx)] = Point2D{
        result.reference_points[static_cast<std::size_t>(seed_idx)].x + result.seed_paramvector[2],
        result.reference_points[static_cast<std::size_t>(seed_idx)].y + result.seed_paramvector[3]};
    result.valid[static_cast<std::size_t>(seed_idx)] = 1;
    queue.push(QueueItem{
        seed_idx,
        AffineParams{
            result.seed_paramvector[2],
            result.seed_paramvector[3],
            result.seed_paramvector[4],
            result.seed_paramvector[5],
            result.seed_paramvector[6],
            result.seed_paramvector[7]},
        result.seed_paramvector[8]});

    const int dxr[4] = {0, 1, 0, -1};
    const int dyr[4] = {-1, 0, 1, 0};
    while (!queue.empty()) {
        const QueueItem item = queue.top();
        queue.pop();
        const Point2D ref = result.reference_points[static_cast<std::size_t>(item.point)];
        const int xr = static_cast<int>(std::lround(ref.x)) / step;
        const int yr = static_cast<int>(std::lround(ref.y)) / step;
        for (int k = 0; k < 4; ++k) {
            const int nxr = xr + dxr[k];
            const int nyr = yr + dyr[k];
            if (nxr < 0 || nyr < 0 || nxr >= prepared.reduced_width || nyr >= prepared.reduced_height) {
                continue;
            }
            const int next_idx = point_index[column_major_index(prepared.reduced_height, nxr, nyr)];
            if (next_idx < 0 || result.valid[static_cast<std::size_t>(next_idx)] != 0) {
                continue;
            }
            const Point2D next_ref = result.reference_points[static_cast<std::size_t>(next_idx)];
            AffineParams init = item.params;
            init.u = item.params.u + item.params.ux * (next_ref.x - ref.x) + item.params.uy * (next_ref.y - ref.y);
            init.v = item.params.v + item.params.vx * (next_ref.x - ref.x) + item.params.vy * (next_ref.y - ref.y);
            AffineParams best_params;
            double best_corr = 0.0;
            const bool found = estimate_affine_near(
                best_params,
                best_corr,
                reference_image,
                deformed_image,
                roi_mask,
                static_cast<int>(std::lround(next_ref.x)),
                static_cast<int>(std::lround(next_ref.y)),
                init,
                config.rg_search_radius,
                config);
            if (!found || best_corr < config.cutoff_corrcoef) {
                continue;
            }
            result.u[static_cast<std::size_t>(next_idx)] = best_params.u;
            result.v[static_cast<std::size_t>(next_idx)] = best_params.v;
            result.ux[static_cast<std::size_t>(next_idx)] = best_params.ux;
            result.uy[static_cast<std::size_t>(next_idx)] = best_params.uy;
            result.vx[static_cast<std::size_t>(next_idx)] = best_params.vx;
            result.vy[static_cast<std::size_t>(next_idx)] = best_params.vy;
            result.corrcoef[static_cast<std::size_t>(next_idx)] = best_corr;
            result.deformed_points[static_cast<std::size_t>(next_idx)] = Point2D{next_ref.x + best_params.u, next_ref.y + best_params.v};
            result.valid[static_cast<std::size_t>(next_idx)] = 1;
            queue.push(QueueItem{next_idx, best_params, best_corr});
        }
    }

    result.ok = true;
    result.message =
        "Reduced-grid DIC2D displacement field computed with no-GUI RG propagation and six-parameter affine IC-GN refinement.";
    return result;
}

}  // namespace multidic::ncorr
