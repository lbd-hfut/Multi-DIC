#include "ncorr_api.h"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

std::map<std::string, std::string> parse_args(const int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::invalid_argument("Arguments must be provided as --key value pairs.");
        }
        args[key.substr(2)] = argv[i + 1];
    }
    return args;
}

const std::string& required_arg(const std::map<std::string, std::string>& args, const std::string& key) {
    const auto iter = args.find(key);
    if (iter == args.end()) {
        throw std::invalid_argument("Missing required argument: --" + key);
    }
    return iter->second;
}

int get_int(const std::map<std::string, std::string>& args, const std::string& key, const int fallback) {
    const auto iter = args.find(key);
    return iter == args.end() ? fallback : std::stoi(iter->second);
}

double get_double(const std::map<std::string, std::string>& args, const std::string& key, const double fallback) {
    const auto iter = args.find(key);
    return iter == args.end() ? fallback : std::stod(iter->second);
}

bool get_bool(const std::map<std::string, std::string>& args, const std::string& key, const bool fallback) {
    const auto iter = args.find(key);
    if (iter == args.end()) {
        return fallback;
    }
    return iter->second == "1" || iter->second == "true" || iter->second == "True";
}

template <typename T>
void read_exact(std::ifstream& stream, T* data, const std::size_t count) {
    stream.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    if (!stream) {
        throw std::runtime_error("Input binary file is truncated.");
    }
}

multidic::ncorr::GrayImage read_gray_image(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not open image binary: " + path);
    }
    std::int32_t width = 0;
    std::int32_t height = 0;
    read_exact(stream, &width, 1);
    read_exact(stream, &height, 1);
    multidic::ncorr::GrayImage image;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.pixels.resize(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height));
    read_exact(stream, image.pixels.data(), image.pixels.size());
    return image;
}

multidic::ncorr::BoolMask read_bool_mask(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not open mask binary: " + path);
    }
    std::int32_t width = 0;
    std::int32_t height = 0;
    read_exact(stream, &width, 1);
    read_exact(stream, &height, 1);
    multidic::ncorr::BoolMask mask;
    mask.width = static_cast<int>(width);
    mask.height = static_cast<int>(height);
    mask.values.resize(static_cast<std::size_t>(mask.width) * static_cast<std::size_t>(mask.height));
    read_exact(stream, mask.values.data(), mask.values.size());
    return mask;
}

void print_json_string(const std::string& text) {
    std::cout << '"';
    for (const char ch : text) {
        if (ch == '"' || ch == '\\') {
            std::cout << '\\' << ch;
        } else if (ch == '\n') {
            std::cout << "\\n";
        } else {
            std::cout << ch;
        }
    }
    std::cout << '"';
}

void write_result_binary(const std::string& path, const multidic::ncorr::NcorrResult& result, const int step) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not open output binary: " + path);
    }
    const std::int32_t width = result.reduced_width;
    const std::int32_t height = result.reduced_height;
    const std::int32_t count = width * height;
    std::vector<double> x_ref(static_cast<std::size_t>(count), 0.0);
    std::vector<double> y_ref(static_cast<std::size_t>(count), 0.0);
    std::vector<double> x_def(static_cast<std::size_t>(count), 0.0);
    std::vector<double> y_def(static_cast<std::size_t>(count), 0.0);
    std::vector<double> u(static_cast<std::size_t>(count), 0.0);
    std::vector<double> v(static_cast<std::size_t>(count), 0.0);
    std::vector<double> ux(static_cast<std::size_t>(count), 0.0);
    std::vector<double> uy(static_cast<std::size_t>(count), 0.0);
    std::vector<double> vx(static_cast<std::size_t>(count), 0.0);
    std::vector<double> vy(static_cast<std::size_t>(count), 0.0);
    std::vector<double> corrcoef(static_cast<std::size_t>(count), 0.0);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(count), 0);
    for (std::size_t i = 0; i < result.reference_points.size(); ++i) {
        const int xr = static_cast<int>(std::lround(result.reference_points[i].x)) / step;
        const int yr = static_cast<int>(std::lround(result.reference_points[i].y)) / step;
        if (xr < 0 || yr < 0 || xr >= width || yr >= height) {
            continue;
        }
        const auto idx = static_cast<std::size_t>(yr) + static_cast<std::size_t>(xr) * static_cast<std::size_t>(height);
        x_ref[idx] = result.reference_points[i].x;
        y_ref[idx] = result.reference_points[i].y;
        x_def[idx] = result.deformed_points[i].x;
        y_def[idx] = result.deformed_points[i].y;
        u[idx] = result.u[i];
        v[idx] = result.v[i];
        ux[idx] = result.ux[i];
        uy[idx] = result.uy[i];
        vx[idx] = result.vx[i];
        vy[idx] = result.vy[i];
        corrcoef[idx] = result.corrcoef[i];
        valid[idx] = result.valid[i];
    }
    stream.write(reinterpret_cast<const char*>(&width), sizeof(width));
    stream.write(reinterpret_cast<const char*>(&height), sizeof(height));
    stream.write(reinterpret_cast<const char*>(x_ref.data()), static_cast<std::streamsize>(sizeof(double) * x_ref.size()));
    stream.write(reinterpret_cast<const char*>(y_ref.data()), static_cast<std::streamsize>(sizeof(double) * y_ref.size()));
    stream.write(reinterpret_cast<const char*>(x_def.data()), static_cast<std::streamsize>(sizeof(double) * x_def.size()));
    stream.write(reinterpret_cast<const char*>(y_def.data()), static_cast<std::streamsize>(sizeof(double) * y_def.size()));
    stream.write(reinterpret_cast<const char*>(u.data()), static_cast<std::streamsize>(sizeof(double) * u.size()));
    stream.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(sizeof(double) * v.size()));
    stream.write(reinterpret_cast<const char*>(ux.data()), static_cast<std::streamsize>(sizeof(double) * ux.size()));
    stream.write(reinterpret_cast<const char*>(uy.data()), static_cast<std::streamsize>(sizeof(double) * uy.size()));
    stream.write(reinterpret_cast<const char*>(vx.data()), static_cast<std::streamsize>(sizeof(double) * vx.size()));
    stream.write(reinterpret_cast<const char*>(vy.data()), static_cast<std::streamsize>(sizeof(double) * vy.size()));
    stream.write(reinterpret_cast<const char*>(corrcoef.data()), static_cast<std::streamsize>(sizeof(double) * corrcoef.size()));
    stream.write(reinterpret_cast<const char*>(valid.data()), static_cast<std::streamsize>(sizeof(std::uint8_t) * valid.size()));
    if (!stream) {
        throw std::runtime_error("Failed while writing output binary: " + path);
    }
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        auto reference = read_gray_image(required_arg(args, "reference"));
        auto deformed = read_gray_image(required_arg(args, "deformed"));
        auto roi = read_bool_mask(required_arg(args, "mask"));

        multidic::ncorr::DIC2DConfig config;
        config.subset_radius = get_int(args, "subset-radius", config.subset_radius);
        config.subset_spacing = get_int(args, "subset-spacing", config.subset_spacing);
        config.seed_search_radius = get_int(args, "seed-search-radius", config.seed_search_radius);
        config.rg_search_radius = get_int(args, "rg-search-radius", config.rg_search_radius);
        config.roi_min_region_area = get_int(args, "roi-min-region-area", config.roi_min_region_area);
        config.cutoff_diffnorm = get_double(args, "cutoff-diffnorm", config.cutoff_diffnorm);
        config.cutoff_iteration = get_int(args, "cutoff-iteration", config.cutoff_iteration);
        config.num_threads = get_int(args, "num-threads", config.num_threads);
        config.subset_truncation = get_bool(args, "subset-truncation", config.subset_truncation);
        config.units_per_pixel = get_double(args, "units-per-pixel", config.units_per_pixel);
        config.cutoff_corrcoef = get_double(args, "cutoff-corrcoef", config.cutoff_corrcoef);
        config.lenscoef = get_double(args, "lenscoef", config.lenscoef);

        multidic::ncorr::SeedPoint seed;
        seed.xy.x = get_double(args, "seed-x", 0.0);
        seed.xy.y = get_double(args, "seed-y", 0.0);
        seed.observation_xy.x = get_double(args, "obs-x", seed.xy.x);
        seed.observation_xy.y = get_double(args, "obs-y", seed.xy.y);

        const auto result = multidic::ncorr::run_ncorr_dic2d(reference, deformed, roi, seed, config);
        const auto output_iter = args.find("output");
        if (output_iter != args.end()) {
            write_result_binary(output_iter->second, result, config.subset_spacing + 1);
        }
        std::cout << "{";
        std::cout << "\"ok\":" << (result.ok ? "true" : "false");
        std::cout << ",\"output_schema_version\":2";
        std::cout << ",\"message\":";
        print_json_string(result.message);
        std::cout << ",\"reduced_width\":" << result.reduced_width;
        std::cout << ",\"reduced_height\":" << result.reduced_height;
        std::cout << ",\"roi_region_count\":" << result.roi_region_count;
        std::cout << ",\"seed_region\":" << result.seed_region;
        std::cout << ",\"seed_paramvector\":[";
        for (std::size_t i = 0; i < result.seed_paramvector.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << result.seed_paramvector[i];
        }
        std::cout << "]";
        std::cout << ",\"reference_grid_points\":" << result.reference_points.size();
        const std::size_t valid_count = std::accumulate(
            result.valid.begin(),
            result.valid.end(),
            std::size_t{0},
            [](const std::size_t total, const std::uint8_t valid) { return total + (valid != 0 ? 1U : 0U); });
        std::cout << ",\"valid_points\":" << valid_count;
        std::cout << "}\n";
        return result.ok ? 0 : 10;
    } catch (const std::exception& exc) {
        std::cout << "{\"ok\":false,\"error\":";
        print_json_string(exc.what());
        std::cout << "}\n";
        return 1;
    }
}
