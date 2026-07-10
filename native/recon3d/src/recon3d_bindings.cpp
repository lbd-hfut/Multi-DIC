#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace py = pybind11;

namespace {

struct Sample {
    double u = 0.0;
    double v = 0.0;
    double corr = 0.0;
    bool ok = false;
};

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

using Mat34 = std::array<double, 12>;
using Mat3 = std::array<double, 9>;

struct PairSurfaceRow {
    std::int64_t point_id = 0;
    Vec2 uv_ref_a{};
    Vec2 uv_ref_b{};
    Vec2 uv_def_a{};
    Vec2 uv_def_b{};
    Vec3 point_ref{};
    Vec3 point_def{};
    double corr_a = 0.0;
    double corr_b = 0.0;
    double corr_comb = 0.0;
    double err_ref = 0.0;
    double err_def = 0.0;
    bool valid = false;
};

double sqr(const double value) {
    return value * value;
}

double dot3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 sub3(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double norm3(const Vec3& a) {
    return std::sqrt(dot3(a, a));
}

Vec3 scale3(const Vec3& a, const double s) {
    return Vec3{a.x * s, a.y * s, a.z * s};
}

bool finite3(const Vec3& a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

Mat3 outer3(const Vec3& a, const Vec3& b) {
    return Mat3{
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z};
}

Mat3 add3(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i = 0; i < 9; ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

Mat3 matmul3(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) {
                out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
            }
        }
    }
    return out;
}

Mat3 transpose3(const Mat3& a) {
    return Mat3{a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
}

double det3(const Mat3& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
           a[2] * (a[3] * a[7] - a[4] * a[6]);
}

bool inverse3(const Mat3& a, Mat3& inv) {
    const double det = det3(a);
    if (std::abs(det) <= 1.0e-18 || !std::isfinite(det)) {
        return false;
    }
    const double s = 1.0 / det;
    inv = Mat3{
        (a[4] * a[8] - a[5] * a[7]) * s,
        (a[2] * a[7] - a[1] * a[8]) * s,
        (a[1] * a[5] - a[2] * a[4]) * s,
        (a[5] * a[6] - a[3] * a[8]) * s,
        (a[0] * a[8] - a[2] * a[6]) * s,
        (a[2] * a[3] - a[0] * a[5]) * s,
        (a[3] * a[7] - a[4] * a[6]) * s,
        (a[1] * a[6] - a[0] * a[7]) * s,
        (a[0] * a[4] - a[1] * a[3]) * s};
    return true;
}

double frobenius3(const Mat3& a) {
    double sum = 0.0;
    for (const double value : a) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

struct Eigen3 {
    std::array<double, 3> values{};
    Mat3 vectors{};
};

Eigen3 jacobi_eigen3(Mat3 A) {
    Mat3 V{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    for (int iter = 0; iter < 64; ++iter) {
        int p = 0;
        int q = 1;
        double max_abs = std::abs(A[1]);
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                const double value = std::abs(A[i * 3 + j]);
                if (value > max_abs) {
                    max_abs = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_abs < 1.0e-14) {
            break;
        }
        const double app = A[p * 3 + p];
        const double aqq = A[q * 3 + q];
        const double apq = A[p * 3 + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double sign = tau >= 0.0 ? 1.0 : -1.0;
        const double t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < 3; ++k) {
            const double aip = A[k * 3 + p];
            const double aiq = A[k * 3 + q];
            A[k * 3 + p] = c * aip - s * aiq;
            A[k * 3 + q] = s * aip + c * aiq;
        }
        for (int k = 0; k < 3; ++k) {
            const double apk = A[p * 3 + k];
            const double aqk = A[q * 3 + k];
            A[p * 3 + k] = c * apk - s * aqk;
            A[q * 3 + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
            const double vip = V[k * 3 + p];
            const double viq = V[k * 3 + q];
            V[k * 3 + p] = c * vip - s * viq;
            V[k * 3 + q] = s * vip + c * viq;
        }
    }

    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](const int a, const int b) { return A[a * 3 + a] < A[b * 3 + b]; });
    Eigen3 out{};
    for (int dst = 0; dst < 3; ++dst) {
        const int src = order[dst];
        out.values[dst] = A[src * 3 + src];
        for (int r = 0; r < 3; ++r) {
            out.vectors[r * 3 + dst] = V[r * 3 + src];
        }
    }
    return out;
}

Sample sample_dic(
    const py::array_t<double>& u_arr,
    const py::array_t<double>& v_arr,
    const py::array_t<double>& corr_arr,
    const py::array_t<bool>& valid_arr,
    const int cam,
    const double px,
    const double py,
    const int reduced_height,
    const int reduced_width,
    const int subset_spacing,
    const double min_corrcoef) {
    const auto u = u_arr.unchecked<3>();
    const auto v = v_arr.unchecked<3>();
    const auto corr = corr_arr.unchecked<3>();
    const auto valid = valid_arr.unchecked<3>();
    const double step = static_cast<double>(subset_spacing + 1);
    const double gx = px / step;
    const double gy = py / step;
    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    if (x0 >= 0 && y0 >= 0 && x1 < reduced_width && y1 < reduced_height &&
        valid(cam, y0, x0) && valid(cam, y0, x1) && valid(cam, y1, x0) && valid(cam, y1, x1)) {
        const double c00 = corr(cam, y0, x0);
        const double c10 = corr(cam, y0, x1);
        const double c01 = corr(cam, y1, x0);
        const double c11 = corr(cam, y1, x1);
        if (std::min(std::min(c00, c10), std::min(c01, c11)) >= min_corrcoef) {
            const double wx = gx - static_cast<double>(x0);
            const double wy = gy - static_cast<double>(y0);
            const double w00 = (1.0 - wx) * (1.0 - wy);
            const double w10 = wx * (1.0 - wy);
            const double w01 = (1.0 - wx) * wy;
            const double w11 = wx * wy;
            return Sample{
                u(cam, y0, x0) * w00 + u(cam, y0, x1) * w10 + u(cam, y1, x0) * w01 + u(cam, y1, x1) * w11,
                v(cam, y0, x0) * w00 + v(cam, y0, x1) * w10 + v(cam, y1, x0) * w01 + v(cam, y1, x1) * w11,
                c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11,
                true};
        }
    }

    const int xn = static_cast<int>(std::llround(gx));
    const int yn = static_cast<int>(std::llround(gy));
    if (xn < 0 || yn < 0 || xn >= reduced_width || yn >= reduced_height ||
        !valid(cam, yn, xn) || corr(cam, yn, xn) < min_corrcoef) {
        return {};
    }
    return Sample{u(cam, yn, xn), v(cam, yn, xn), corr(cam, yn, xn), true};
}

Vec2 pixel_to_normalized(
    const py::detail::unchecked_reference<double, 3>& K,
    const py::detail::unchecked_reference<double, 2>& dist,
    const int cam,
    const double px,
    const double py) {
    double x = (px - K(cam, 0, 2)) / K(cam, 0, 0);
    double y = (py - K(cam, 1, 2)) / K(cam, 1, 1);
    const double k1 = dist.shape(1) > 0 ? dist(cam, 0) : 0.0;
    const double k2 = dist.shape(1) > 1 ? dist(cam, 1) : 0.0;
    if (std::abs(k1) > 1.0e-12 || std::abs(k2) > 1.0e-12) {
        double xu = x;
        double yu = y;
        for (int i = 0; i < 8; ++i) {
            const double r2 = xu * xu + yu * yu;
            const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            if (std::abs(radial) <= 1.0e-12) {
                break;
            }
            xu = x / radial;
            yu = y / radial;
        }
        x = xu;
        y = yu;
    }
    return Vec2{x, y};
}

Mat34 projection_rt(
    const py::detail::unchecked_reference<double, 3>& R,
    const py::detail::unchecked_reference<double, 2>& t,
    const int cam) {
    return Mat34{
        R(cam, 0, 0), R(cam, 0, 1), R(cam, 0, 2), t(cam, 0),
        R(cam, 1, 0), R(cam, 1, 1), R(cam, 1, 2), t(cam, 1),
        R(cam, 2, 0), R(cam, 2, 1), R(cam, 2, 2), t(cam, 2)};
}

bool jacobi_eigen_smallest(std::array<double, 16> A, std::array<double, 4>& vec) {
    std::array<double, 16> V{};
    for (int i = 0; i < 4; ++i) {
        V[i * 4 + i] = 1.0;
    }
    for (int iter = 0; iter < 64; ++iter) {
        int p = 0;
        int q = 1;
        double max_abs = std::abs(A[p * 4 + q]);
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                const double value = std::abs(A[i * 4 + j]);
                if (value > max_abs) {
                    max_abs = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_abs < 1.0e-12) {
            break;
        }
        const double app = A[p * 4 + p];
        const double aqq = A[q * 4 + q];
        const double apq = A[p * 4 + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double sign = tau >= 0.0 ? 1.0 : -1.0;
        const double tt = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + tt * tt);
        const double s = tt * c;
        for (int k = 0; k < 4; ++k) {
            const double aik = A[p * 4 + k];
            const double aqk = A[q * 4 + k];
            A[p * 4 + k] = c * aik - s * aqk;
            A[q * 4 + k] = s * aik + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
            const double akp = A[k * 4 + p];
            const double akq = A[k * 4 + q];
            A[k * 4 + p] = c * akp - s * akq;
            A[k * 4 + q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
            const double vip = V[k * 4 + p];
            const double viq = V[k * 4 + q];
            V[k * 4 + p] = c * vip - s * viq;
            V[k * 4 + q] = s * vip + c * viq;
        }
    }
    int min_idx = 0;
    double min_eval = A[0];
    for (int i = 1; i < 4; ++i) {
        if (A[i * 4 + i] < min_eval) {
            min_eval = A[i * 4 + i];
            min_idx = i;
        }
    }
    for (int i = 0; i < 4; ++i) {
        vec[static_cast<std::size_t>(i)] = V[i * 4 + min_idx];
    }
    return std::abs(vec[3]) > 1.0e-12;
}

bool triangulate(const std::vector<Vec2>& rays, const std::vector<Mat34>& P, Vec3& out) {
    if (rays.size() < 2) {
        return false;
    }
    std::array<double, 16> AtA{};
    for (std::size_t i = 0; i < rays.size(); ++i) {
        std::array<double, 4> r1{
            rays[i].x * P[i][8] - P[i][0],
            rays[i].x * P[i][9] - P[i][1],
            rays[i].x * P[i][10] - P[i][2],
            rays[i].x * P[i][11] - P[i][3]};
        std::array<double, 4> r2{
            rays[i].y * P[i][8] - P[i][4],
            rays[i].y * P[i][9] - P[i][5],
            rays[i].y * P[i][10] - P[i][6],
            rays[i].y * P[i][11] - P[i][7]};
        for (const auto& row : {r1, r2}) {
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    AtA[a * 4 + b] += row[a] * row[b];
                }
            }
        }
    }
    std::array<double, 4> homog{};
    if (!jacobi_eigen_smallest(AtA, homog)) {
        return false;
    }
    out = Vec3{homog[0] / homog[3], homog[1] / homog[3], homog[2] / homog[3]};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z);
}

Vec2 project(
    const Vec3& point,
    const py::detail::unchecked_reference<double, 3>& K,
    const py::detail::unchecked_reference<double, 2>& dist,
    const py::detail::unchecked_reference<double, 3>& R,
    const py::detail::unchecked_reference<double, 2>& t,
    const int cam) {
    const double X = R(cam, 0, 0) * point.x + R(cam, 0, 1) * point.y + R(cam, 0, 2) * point.z + t(cam, 0);
    const double Y = R(cam, 1, 0) * point.x + R(cam, 1, 1) * point.y + R(cam, 1, 2) * point.z + t(cam, 1);
    const double Z = R(cam, 2, 0) * point.x + R(cam, 2, 1) * point.y + R(cam, 2, 2) * point.z + t(cam, 2);
    double x = X / Z;
    double y = Y / Z;
    const double k1 = dist.shape(1) > 0 ? dist(cam, 0) : 0.0;
    const double k2 = dist.shape(1) > 1 ? dist(cam, 1) : 0.0;
    if (std::abs(k1) > 1.0e-12 || std::abs(k2) > 1.0e-12) {
        const double r2 = x * x + y * y;
        const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        x *= radial;
        y *= radial;
    }
    return Vec2{K(cam, 0, 0) * x + K(cam, 0, 2), K(cam, 1, 1) * y + K(cam, 1, 2)};
}

py::dict reconstruct_tracks(
    py::array_t<double> K_arr,
    py::array_t<double> dist_arr,
    py::array_t<double> R_arr,
    py::array_t<double> t_arr,
    py::array_t<std::int64_t> point_indices_arr,
    py::array_t<std::int32_t> cam_indices_arr,
    py::array_t<double> uv_arr,
    py::array_t<double> dic_u,
    py::array_t<double> dic_v,
    py::array_t<double> dic_corrcoef,
    py::array_t<bool> dic_valid,
    const int reduced_height,
    const int reduced_width,
    const int subset_spacing,
    const int min_views,
    const double min_corrcoef,
    const double max_reprojection_error_px,
    const double scale) {
    const auto K = K_arr.unchecked<3>();
    const auto dist = dist_arr.unchecked<2>();
    const auto R = R_arr.unchecked<3>();
    const auto t = t_arr.unchecked<2>();
    const auto point_indices = point_indices_arr.unchecked<1>();
    const auto cam_indices = cam_indices_arr.unchecked<1>();
    const auto uv = uv_arr.unchecked<2>();

    std::map<std::int64_t, std::vector<py::ssize_t>> by_track;
    for (py::ssize_t i = 0; i < point_indices.shape(0); ++i) {
        by_track[point_indices(i)].push_back(i);
    }
    const py::ssize_t n = static_cast<py::ssize_t>(by_track.size());
    py::array_t<std::int64_t> out_ids(n);
    py::array_t<double> points_ref({n, py::ssize_t{3}});
    py::array_t<double> points_def({n, py::ssize_t{3}});
    py::array_t<double> disp({n, py::ssize_t{3}});
    py::array_t<double> points_ref_world({n, py::ssize_t{3}});
    py::array_t<double> points_def_world({n, py::ssize_t{3}});
    py::array_t<double> disp_world({n, py::ssize_t{3}});
    py::array_t<std::int32_t> num_views(n);
    py::array_t<double> mean_corr(n);
    py::array_t<double> err_ref(n);
    py::array_t<double> err_def(n);
    py::array_t<bool> valid(n);

    auto ids_o = out_ids.mutable_unchecked<1>();
    auto pr = points_ref.mutable_unchecked<2>();
    auto pd = points_def.mutable_unchecked<2>();
    auto dd = disp.mutable_unchecked<2>();
    auto prw = points_ref_world.mutable_unchecked<2>();
    auto pdw = points_def_world.mutable_unchecked<2>();
    auto dww = disp_world.mutable_unchecked<2>();
    auto nv = num_views.mutable_unchecked<1>();
    auto mc = mean_corr.mutable_unchecked<1>();
    auto er = err_ref.mutable_unchecked<1>();
    auto ed = err_def.mutable_unchecked<1>();
    auto va = valid.mutable_unchecked<1>();

    py::ssize_t out = 0;
    for (const auto& [track_id, obs_ids] : by_track) {
        ids_o(out) = track_id;
        Vec3 Xr{};
        Vec3 Xd{};
        std::vector<Vec2> rays_ref;
        std::vector<Vec2> rays_def;
        std::vector<Mat34> projections;
        std::vector<int> cams;
        std::vector<Vec2> uv_ref_used;
        std::vector<Vec2> uv_def_used;
        double corr_sum = 0.0;
        for (const py::ssize_t obs_id : obs_ids) {
            const int cam = cam_indices(obs_id);
            const double px = uv(obs_id, 0);
            const double pyv = uv(obs_id, 1);
            const Sample sample = sample_dic(dic_u, dic_v, dic_corrcoef, dic_valid, cam, px, pyv, reduced_height, reduced_width, subset_spacing, min_corrcoef);
            if (!sample.ok) {
                continue;
            }
            rays_ref.push_back(pixel_to_normalized(K, dist, cam, px, pyv));
            rays_def.push_back(pixel_to_normalized(K, dist, cam, px + sample.u, pyv + sample.v));
            projections.push_back(projection_rt(R, t, cam));
            cams.push_back(cam);
            uv_ref_used.push_back(Vec2{px, pyv});
            uv_def_used.push_back(Vec2{px + sample.u, pyv + sample.v});
            corr_sum += sample.corr;
        }
        bool ok = static_cast<int>(cams.size()) >= min_views && triangulate(rays_ref, projections, Xr) && triangulate(rays_def, projections, Xd);
        double ref_error = std::numeric_limits<double>::infinity();
        double def_error = std::numeric_limits<double>::infinity();
        if (ok) {
            ref_error = 0.0;
            def_error = 0.0;
            for (std::size_t i = 0; i < cams.size(); ++i) {
                const Vec2 rr = project(Xr, K, dist, R, t, cams[i]);
                const Vec2 ddp = project(Xd, K, dist, R, t, cams[i]);
                ref_error += std::sqrt(sqr(rr.x - uv_ref_used[i].x) + sqr(rr.y - uv_ref_used[i].y));
                def_error += std::sqrt(sqr(ddp.x - uv_def_used[i].x) + sqr(ddp.y - uv_def_used[i].y));
            }
            ref_error /= static_cast<double>(cams.size());
            def_error /= static_cast<double>(cams.size());
            ok = ref_error <= max_reprojection_error_px && def_error <= max_reprojection_error_px;
        }
        const std::array<double, 3> ref{Xr.x, Xr.y, Xr.z};
        const std::array<double, 3> def{Xd.x, Xd.y, Xd.z};
        for (int j = 0; j < 3; ++j) {
            pr(out, j) = ok ? ref[j] : 0.0;
            pd(out, j) = ok ? def[j] : 0.0;
            dd(out, j) = ok ? def[j] - ref[j] : 0.0;
            prw(out, j) = pr(out, j) * scale;
            pdw(out, j) = pd(out, j) * scale;
            dww(out, j) = dd(out, j) * scale;
        }
        nv(out) = static_cast<std::int32_t>(cams.size());
        mc(out) = cams.empty() ? 0.0 : corr_sum / static_cast<double>(cams.size());
        er(out) = ref_error;
        ed(out) = def_error;
        va(out) = ok;
        ++out;
    }

    py::dict result;
    result["point_indices"] = out_ids;
    result["points_ref_sfm"] = points_ref;
    result["points_def_sfm"] = points_def;
    result["displacement_sfm"] = disp;
    result["points_ref_world"] = points_ref_world;
    result["points_def_world"] = points_def_world;
    result["displacement_world"] = disp_world;
    result["num_views"] = num_views;
    result["mean_corrcoef"] = mean_corr;
    result["reprojection_error_ref"] = err_ref;
    result["reprojection_error_def"] = err_def;
    result["valid"] = valid;
    return result;
}

py::dict reconstruct_pair_surface_points(
    py::array_t<double> K_arr,
    py::array_t<double> dist_arr,
    py::array_t<double> R_arr,
    py::array_t<double> t_arr,
    py::array_t<std::int64_t> point_indices_arr,
    py::array_t<std::int32_t> cam_indices_arr,
    py::array_t<double> uv_arr,
    py::array_t<double> dic_u,
    py::array_t<double> dic_v,
    py::array_t<double> dic_corrcoef,
    py::array_t<bool> dic_valid,
    const int reduced_height,
    const int reduced_width,
    const int subset_spacing,
    const int cam_a,
    const int cam_b,
    const double min_corrcoef,
    const double max_reprojection_error_px,
    const double scale) {
    const auto K = K_arr.unchecked<3>();
    const auto dist = dist_arr.unchecked<2>();
    const auto R = R_arr.unchecked<3>();
    const auto t = t_arr.unchecked<2>();
    const auto point_indices = point_indices_arr.unchecked<1>();
    const auto cam_indices = cam_indices_arr.unchecked<1>();
    const auto uv = uv_arr.unchecked<2>();

    std::map<std::int64_t, std::array<py::ssize_t, 2>> by_pair;
    for (py::ssize_t obs = 0; obs < point_indices.shape(0); ++obs) {
        const int cam = cam_indices(obs);
        if (cam != cam_a && cam != cam_b) {
            continue;
        }
        auto& slots = by_pair[point_indices(obs)];
        if (slots[0] == 0 && slots[1] == 0) {
            slots = std::array<py::ssize_t, 2>{-1, -1};
        }
        slots[cam == cam_a ? 0 : 1] = obs;
    }

    std::vector<PairSurfaceRow> rows;
    rows.reserve(by_pair.size());
    for (const auto& [point_id, obs_pair] : by_pair) {
        const py::ssize_t obs_a = obs_pair[0];
        const py::ssize_t obs_b = obs_pair[1];
        if (obs_a < 0 || obs_b < 0) {
            continue;
        }
        const double ax = uv(obs_a, 0);
        const double ay = uv(obs_a, 1);
        const double bx = uv(obs_b, 0);
        const double by = uv(obs_b, 1);
        const Sample sample_a = sample_dic(dic_u, dic_v, dic_corrcoef, dic_valid, cam_a, ax, ay, reduced_height, reduced_width, subset_spacing, min_corrcoef);
        const Sample sample_b = sample_dic(dic_u, dic_v, dic_corrcoef, dic_valid, cam_b, bx, by, reduced_height, reduced_width, subset_spacing, min_corrcoef);
        if (!sample_a.ok || !sample_b.ok) {
            continue;
        }
        const Vec2 uv_ref_a{ax, ay};
        const Vec2 uv_ref_b{bx, by};
        const Vec2 uv_def_a{ax + sample_a.u, ay + sample_a.v};
        const Vec2 uv_def_b{bx + sample_b.u, by + sample_b.v};
        std::vector<Vec2> rays_ref{
            pixel_to_normalized(K, dist, cam_a, uv_ref_a.x, uv_ref_a.y),
            pixel_to_normalized(K, dist, cam_b, uv_ref_b.x, uv_ref_b.y)};
        std::vector<Vec2> rays_def{
            pixel_to_normalized(K, dist, cam_a, uv_def_a.x, uv_def_a.y),
            pixel_to_normalized(K, dist, cam_b, uv_def_b.x, uv_def_b.y)};
        std::vector<Mat34> projections{projection_rt(R, t, cam_a), projection_rt(R, t, cam_b)};
        Vec3 Xr{};
        Vec3 Xd{};
        if (!triangulate(rays_ref, projections, Xr) || !triangulate(rays_def, projections, Xd)) {
            continue;
        }
        const Vec2 ref_proj_a = project(Xr, K, dist, R, t, cam_a);
        const Vec2 ref_proj_b = project(Xr, K, dist, R, t, cam_b);
        const Vec2 def_proj_a = project(Xd, K, dist, R, t, cam_a);
        const Vec2 def_proj_b = project(Xd, K, dist, R, t, cam_b);
        const double err_ref = 0.5 * (
            std::sqrt(sqr(ref_proj_a.x - uv_ref_a.x) + sqr(ref_proj_a.y - uv_ref_a.y)) +
            std::sqrt(sqr(ref_proj_b.x - uv_ref_b.x) + sqr(ref_proj_b.y - uv_ref_b.y)));
        const double err_def = 0.5 * (
            std::sqrt(sqr(def_proj_a.x - uv_def_a.x) + sqr(def_proj_a.y - uv_def_a.y)) +
            std::sqrt(sqr(def_proj_b.x - uv_def_b.x) + sqr(def_proj_b.y - uv_def_b.y)));
        const double corr_comb = std::min(sample_a.corr, sample_b.corr);
        rows.push_back(PairSurfaceRow{
            point_id,
            uv_ref_a,
            uv_ref_b,
            uv_def_a,
            uv_def_b,
            Xr,
            Xd,
            sample_a.corr,
            sample_b.corr,
            corr_comb,
            err_ref,
            err_def,
            corr_comb >= min_corrcoef && err_ref <= max_reprojection_error_px && err_def <= max_reprojection_error_px});
    }

    const py::ssize_t n = static_cast<py::ssize_t>(rows.size());
    py::array_t<std::int64_t> point_ids(n);
    py::array_t<double> uv_ref_a({n, py::ssize_t{2}});
    py::array_t<double> uv_ref_b({n, py::ssize_t{2}});
    py::array_t<double> uv_def_a({n, py::ssize_t{2}});
    py::array_t<double> uv_def_b({n, py::ssize_t{2}});
    py::array_t<double> points_ref({n, py::ssize_t{3}});
    py::array_t<double> points_def({n, py::ssize_t{3}});
    py::array_t<double> points_ref_world({n, py::ssize_t{3}});
    py::array_t<double> points_def_world({n, py::ssize_t{3}});
    py::array_t<double> displacement({n, py::ssize_t{3}});
    py::array_t<double> displacement_world({n, py::ssize_t{3}});
    py::array_t<double> displacement_norm_world(n);
    py::array_t<double> corr_a(n);
    py::array_t<double> corr_b(n);
    py::array_t<double> corr_comb(n);
    py::array_t<double> reproj_ref(n);
    py::array_t<double> reproj_def(n);
    py::array_t<bool> valid_points(n);

    auto ids = point_ids.mutable_unchecked<1>();
    auto ura = uv_ref_a.mutable_unchecked<2>();
    auto urb = uv_ref_b.mutable_unchecked<2>();
    auto uda = uv_def_a.mutable_unchecked<2>();
    auto udb = uv_def_b.mutable_unchecked<2>();
    auto pr = points_ref.mutable_unchecked<2>();
    auto pd = points_def.mutable_unchecked<2>();
    auto prw = points_ref_world.mutable_unchecked<2>();
    auto pdw = points_def_world.mutable_unchecked<2>();
    auto disp = displacement.mutable_unchecked<2>();
    auto dispw = displacement_world.mutable_unchecked<2>();
    auto dispn = displacement_norm_world.mutable_unchecked<1>();
    auto ca = corr_a.mutable_unchecked<1>();
    auto cb = corr_b.mutable_unchecked<1>();
    auto cc = corr_comb.mutable_unchecked<1>();
    auto er = reproj_ref.mutable_unchecked<1>();
    auto ed = reproj_def.mutable_unchecked<1>();
    auto vp = valid_points.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        const PairSurfaceRow& row = rows[static_cast<std::size_t>(i)];
        ids(i) = row.point_id;
        ura(i, 0) = row.uv_ref_a.x;
        ura(i, 1) = row.uv_ref_a.y;
        urb(i, 0) = row.uv_ref_b.x;
        urb(i, 1) = row.uv_ref_b.y;
        uda(i, 0) = row.uv_def_a.x;
        uda(i, 1) = row.uv_def_a.y;
        udb(i, 0) = row.uv_def_b.x;
        udb(i, 1) = row.uv_def_b.y;
        const std::array<double, 3> ref{row.point_ref.x, row.point_ref.y, row.point_ref.z};
        const std::array<double, 3> def{row.point_def.x, row.point_def.y, row.point_def.z};
        double norm2 = 0.0;
        for (int j = 0; j < 3; ++j) {
            const double d = def[j] - ref[j];
            pr(i, j) = ref[j];
            pd(i, j) = def[j];
            prw(i, j) = ref[j] * scale;
            pdw(i, j) = def[j] * scale;
            disp(i, j) = d;
            dispw(i, j) = d * scale;
            norm2 += sqr(d * scale);
        }
        dispn(i) = std::sqrt(norm2);
        ca(i) = row.corr_a;
        cb(i) = row.corr_b;
        cc(i) = row.corr_comb;
        er(i) = row.err_ref;
        ed(i) = row.err_def;
        vp(i) = row.valid;
    }

    py::dict result;
    result["point_indices"] = point_ids;
    result["uv_ref_a"] = uv_ref_a;
    result["uv_ref_b"] = uv_ref_b;
    result["uv_def_a"] = uv_def_a;
    result["uv_def_b"] = uv_def_b;
    result["points_ref_sfm"] = points_ref;
    result["points_def_sfm"] = points_def;
    result["points_ref_world"] = points_ref_world;
    result["points_def_world"] = points_def_world;
    result["displacement_sfm"] = displacement;
    result["displacement_world"] = displacement_world;
    result["displacement_norm_world"] = displacement_norm_world;
    result["corr_a"] = corr_a;
    result["corr_b"] = corr_b;
    result["corr_comb"] = corr_comb;
    result["reprojection_error_ref"] = reproj_ref;
    result["reprojection_error_def"] = reproj_def;
    result["valid_points"] = valid_points;
    return result;
}

py::dict compute_surface_deformation(
    py::array_t<std::int32_t> faces_arr,
    py::array_t<double> points_ref_arr,
    py::array_t<double> points_def_arr,
    py::array_t<bool> valid_faces_arr) {
    const auto faces = faces_arr.unchecked<2>();
    const auto pref = points_ref_arr.unchecked<2>();
    const auto pdef = points_def_arr.unchecked<2>();
    const auto valid_faces = valid_faces_arr.unchecked<1>();
    const py::ssize_t n = faces.shape(0);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    py::array_t<double> Fmat({n, py::ssize_t{3}, py::ssize_t{3}});
    py::array_t<double> Cmat({n, py::ssize_t{3}, py::ssize_t{3}});
    py::array_t<double> Emat({n, py::ssize_t{3}, py::ssize_t{3}});
    py::array_t<double> emat({n, py::ssize_t{3}, py::ssize_t{3}});
    py::array_t<double> d3_arr({n, py::ssize_t{3}});
    py::array_t<double> J(n);
    py::array_t<double> Emgn(n);
    py::array_t<double> emgn(n);
    py::array_t<double> Epc1(n);
    py::array_t<double> Epc2(n);
    py::array_t<double> epc1(n);
    py::array_t<double> epc2(n);
    py::array_t<double> EShearMax(n);
    py::array_t<double> eShearMax(n);
    py::array_t<double> Eeq(n);
    py::array_t<double> eeq(n);
    py::array_t<double> Area(n);
    py::array_t<double> Lambda1(n);
    py::array_t<double> Lambda2(n);
    py::array_t<bool> valid_out(n);

    auto Fo = Fmat.mutable_unchecked<3>();
    auto Co = Cmat.mutable_unchecked<3>();
    auto Eo = Emat.mutable_unchecked<3>();
    auto eo = emat.mutable_unchecked<3>();
    auto d3o = d3_arr.mutable_unchecked<2>();
    auto Jo = J.mutable_unchecked<1>();
    auto Emo = Emgn.mutable_unchecked<1>();
    auto emo = emgn.mutable_unchecked<1>();
    auto E1o = Epc1.mutable_unchecked<1>();
    auto E2o = Epc2.mutable_unchecked<1>();
    auto e1o = epc1.mutable_unchecked<1>();
    auto e2o = epc2.mutable_unchecked<1>();
    auto Eso = EShearMax.mutable_unchecked<1>();
    auto eso = eShearMax.mutable_unchecked<1>();
    auto Eeqo = Eeq.mutable_unchecked<1>();
    auto eeqo = eeq.mutable_unchecked<1>();
    auto Areao = Area.mutable_unchecked<1>();
    auto L1o = Lambda1.mutable_unchecked<1>();
    auto L2o = Lambda2.mutable_unchecked<1>();
    auto vo = valid_out.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        Jo(i) = Emo(i) = emo(i) = E1o(i) = E2o(i) = e1o(i) = e2o(i) = Eso(i) = eso(i) = Eeqo(i) = eeqo(i) = Areao(i) = L1o(i) = L2o(i) = nan;
        vo(i) = false;
        for (int r = 0; r < 3; ++r) {
            d3o(i, r) = nan;
            for (int c = 0; c < 3; ++c) {
                Fo(i, r, c) = Co(i, r, c) = Eo(i, r, c) = eo(i, r, c) = nan;
            }
        }
        if (!valid_faces(i)) {
            continue;
        }
        const int a = faces(i, 0);
        const int b = faces(i, 1);
        const int cidx = faces(i, 2);
        Vec3 X1{pref(a, 0), pref(a, 1), pref(a, 2)};
        Vec3 X2{pref(b, 0), pref(b, 1), pref(b, 2)};
        Vec3 X3{pref(cidx, 0), pref(cidx, 1), pref(cidx, 2)};
        Vec3 x1{pdef(a, 0), pdef(a, 1), pdef(a, 2)};
        Vec3 x2{pdef(b, 0), pdef(b, 1), pdef(b, 2)};
        Vec3 x3{pdef(cidx, 0), pdef(cidx, 1), pdef(cidx, 2)};
        if (!finite3(X1) || !finite3(X2) || !finite3(X3) || !finite3(x1) || !finite3(x2) || !finite3(x3)) {
            continue;
        }

        const Vec3 D1 = sub3(X2, X1);
        const Vec3 D2 = sub3(X3, X1);
        const Vec3 d1 = sub3(x2, x1);
        const Vec3 d2 = sub3(x3, x1);
        const Vec3 crossD = cross3(D1, D2);
        const Vec3 crossd = cross3(d1, d2);
        const double normD = norm3(crossD);
        const double normd = norm3(crossd);
        if (normD <= 1.0e-18 || normd <= 1.0e-18) {
            continue;
        }
        const Vec3 D3 = scale3(crossD, 1.0 / normD);
        const Vec3 d3 = scale3(crossd, 1.0 / normd);
        const double Dnorm = dot3(crossD, D3);
        if (std::abs(Dnorm) <= 1.0e-18) {
            continue;
        }
        const Vec3 Drec1 = scale3(cross3(D2, D3), 1.0 / Dnorm);
        const Vec3 Drec2 = scale3(cross3(D3, D1), 1.0 / Dnorm);
        const Mat3 F = add3(add3(outer3(d1, Drec1), outer3(d2, Drec2)), outer3(d3, D3));
        const Mat3 C = matmul3(transpose3(F), F);
        Mat3 B_inv{};
        if (!inverse3(matmul3(F, transpose3(F)), B_inv)) {
            continue;
        }

        Mat3 E{};
        Mat3 e{};
        for (int k = 0; k < 9; ++k) {
            const double identity = (k == 0 || k == 4 || k == 8) ? 1.0 : 0.0;
            E[k] = 0.5 * (C[k] - identity);
            e[k] = 0.5 * (identity - B_inv[k]);
        }

        const Eigen3 Ceig = jacobi_eigen3(C);
        std::array<double, 3> lambdas{
            std::sqrt(std::max(0.0, Ceig.values[0])),
            std::sqrt(std::max(0.0, Ceig.values[1])),
            std::sqrt(std::max(0.0, Ceig.values[2]))};
        int remove_lambda = 0;
        double closest = std::abs(lambdas[0] - 1.0);
        for (int k = 1; k < 3; ++k) {
            const double diff = std::abs(lambdas[k] - 1.0);
            if (diff < closest) {
                closest = diff;
                remove_lambda = k;
            }
        }
        std::array<double, 2> planar_lambda{};
        int li = 0;
        for (int k = 0; k < 3; ++k) {
            if (k != remove_lambda) {
                planar_lambda[li++] = lambdas[k];
            }
        }
        std::sort(planar_lambda.begin(), planar_lambda.end());

        const Eigen3 Eeig = jacobi_eigen3(E);
        const Eigen3 eeig = jacobi_eigen3(e);
        auto planar_values = [](const Eigen3& eig, const Vec3& normal) {
            int remove = 0;
            double best = -1.0;
            for (int k = 0; k < 3; ++k) {
                const Vec3 v{eig.vectors[k], eig.vectors[3 + k], eig.vectors[6 + k]};
                const double alignment = std::abs(dot3(v, normal));
                if (alignment > best) {
                    best = alignment;
                    remove = k;
                }
            }
            std::array<double, 2> vals{};
            int out = 0;
            for (int k = 0; k < 3; ++k) {
                if (k != remove) {
                    vals[out++] = eig.values[k];
                }
            }
            std::sort(vals.begin(), vals.end());
            return vals;
        };
        const std::array<double, 2> Eplanar = planar_values(Eeig, D3);
        const std::array<double, 2> eplanar = planar_values(eeig, d3);
        const double traceE = E[0] + E[4] + E[8];
        const double tracee = e[0] + e[4] + e[8];
        Mat3 Edev = E;
        Mat3 edev = e;
        Edev[0] -= traceE / 3.0;
        Edev[4] -= traceE / 3.0;
        Edev[8] -= traceE / 3.0;
        edev[0] -= tracee / 3.0;
        edev[4] -= tracee / 3.0;
        edev[8] -= tracee / 3.0;

        for (int r = 0; r < 3; ++r) {
            d3o(i, r) = r == 0 ? d3.x : (r == 1 ? d3.y : d3.z);
            for (int c = 0; c < 3; ++c) {
                Fo(i, r, c) = F[r * 3 + c];
                Co(i, r, c) = C[r * 3 + c];
                Eo(i, r, c) = E[r * 3 + c];
                eo(i, r, c) = e[r * 3 + c];
            }
        }
        Jo(i) = det3(F);
        Emo(i) = frobenius3(E);
        emo(i) = frobenius3(e);
        E1o(i) = Eplanar[0];
        E2o(i) = Eplanar[1];
        e1o(i) = eplanar[0];
        e2o(i) = eplanar[1];
        Eso(i) = 0.5 * (Eplanar[1] - Eplanar[0]);
        eso(i) = 0.5 * (eplanar[1] - eplanar[0]);
        Eeqo(i) = std::sqrt((2.0 / 3.0) * frobenius3(Edev) * frobenius3(Edev));
        eeqo(i) = std::sqrt((2.0 / 3.0) * frobenius3(edev) * frobenius3(edev));
        Areao(i) = 0.5 * Dnorm;
        L1o(i) = planar_lambda[0];
        L2o(i) = planar_lambda[1];
        vo(i) = true;
    }

    py::dict result;
    result["Fmat"] = Fmat;
    result["Cmat"] = Cmat;
    result["J"] = J;
    result["Emat"] = Emat;
    result["emat"] = emat;
    result["Emgn"] = Emgn;
    result["emgn"] = emgn;
    result["Epc1"] = Epc1;
    result["Epc2"] = Epc2;
    result["epc1"] = epc1;
    result["epc2"] = epc2;
    result["EShearMax"] = EShearMax;
    result["eShearMax"] = eShearMax;
    result["Eeq"] = Eeq;
    result["eeq"] = eeq;
    result["Area"] = Area;
    result["d3"] = d3_arr;
    result["Lambda1"] = Lambda1;
    result["Lambda2"] = Lambda2;
    result["valid_strain_faces"] = valid_out;
    return result;
}

}  // namespace

PYBIND11_MODULE(native_recon3d, m) {
    m.doc() = "Native acceleration kernels for Multi-DIC recon3d.";
    m.def("reconstruct_tracks", &reconstruct_tracks, py::arg("K"), py::arg("dist"), py::arg("R"), py::arg("t"),
          py::arg("point_indices"), py::arg("cam_indices"), py::arg("uv"), py::arg("dic_u"), py::arg("dic_v"),
          py::arg("dic_corrcoef"), py::arg("dic_valid"), py::arg("reduced_height"), py::arg("reduced_width"),
          py::arg("subset_spacing"), py::arg("min_views"), py::arg("min_corrcoef"),
          py::arg("max_reprojection_error_px"), py::arg("scale"));
    m.def("reconstruct_pair_surface_points", &reconstruct_pair_surface_points, py::arg("K"), py::arg("dist"),
          py::arg("R"), py::arg("t"), py::arg("point_indices"), py::arg("cam_indices"), py::arg("uv"),
          py::arg("dic_u"), py::arg("dic_v"), py::arg("dic_corrcoef"), py::arg("dic_valid"),
          py::arg("reduced_height"), py::arg("reduced_width"), py::arg("subset_spacing"), py::arg("cam_a"),
          py::arg("cam_b"), py::arg("min_corrcoef"), py::arg("max_reprojection_error_px"), py::arg("scale"));
    m.def("compute_surface_deformation", &compute_surface_deformation, py::arg("faces"), py::arg("points_ref"),
          py::arg("points_def"), py::arg("valid_faces"));
}
