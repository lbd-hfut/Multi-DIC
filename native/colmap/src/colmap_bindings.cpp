#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MDIC_WITH_EMBEDDED_COLMAP
#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/pairing.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/util/file.h"
#endif

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string option_value(const py::dict& options, const char* key, const std::string& fallback) {
    if (!options.contains(key) || options[py::str(key)].is_none()) {
        return fallback;
    }
    return py::str(options[py::str(key)]);
}

int option_int(const py::dict& options, const char* key, const int fallback) {
    if (!options.contains(key) || options[py::str(key)].is_none()) {
        return fallback;
    }
    return py::cast<int>(options[py::str(key)]);
}

double option_double(const py::dict& options, const char* key, const double fallback) {
    if (!options.contains(key) || options[py::str(key)].is_none()) {
        return fallback;
    }
    return py::cast<double>(options[py::str(key)]);
}

bool option_bool(const py::dict& options, const char* key, const bool fallback) {
    if (!options.contains(key) || options[py::str(key)].is_none()) {
        return fallback;
    }
    return py::cast<bool>(options[py::str(key)]);
}

std::vector<std::pair<std::string, std::string>> ring_image_pairs(
    const std::vector<std::string>& image_names,
    const int window,
    const bool wrap
) {
    std::vector<std::pair<std::string, std::string>> pairs;
    if (image_names.size() < 2 || window < 1) {
        return pairs;
    }
    const int count = static_cast<int>(image_names.size());
    for (int idx = 0; idx < count; ++idx) {
        for (int step = 1; step <= window; ++step) {
            const int next = idx + step;
            if (next < count) {
                pairs.emplace_back(image_names[static_cast<size_t>(idx)], image_names[static_cast<size_t>(next)]);
            } else if (wrap && count > 2) {
                pairs.emplace_back(image_names[static_cast<size_t>(idx)], image_names[static_cast<size_t>(next % count)]);
            }
        }
    }
    return pairs;
}

fs::path write_image_pair_list(
    const fs::path& database_path,
    const std::vector<std::pair<std::string, std::string>>& pairs
) {
    const fs::path pair_path = fs::path(database_path).parent_path() / "image_pairs.txt";
    std::ofstream file(pair_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not write COLMAP image pair list: " + pair_path.string());
    }
    for (const auto& pair : pairs) {
        file << pair.first << " " << pair.second << "\n";
    }
    return pair_path;
}

py::dict step_record(const std::string& name, const std::string& backend) {
    py::dict record;
    record["name"] = name;
    record["backend"] = backend;
    return record;
}

void run_command(const std::string& command, const std::string& name, const fs::path& log_dir, py::list& command_logs) {
    fs::create_directories(log_dir);
    const fs::path log_path = log_dir / (name + ".log");
    const std::string redirected = command + " > " + shell_quote(log_path.string()) + " 2>&1";

    py::dict log_record;
    log_record["name"] = name;
    log_record["log_path"] = log_path.string();
    log_record["command"] = command;

    const int code = std::system(redirected.c_str());
    log_record["exit_code"] = code;
    command_logs.append(log_record);
    if (code != 0) {
        std::ostringstream message;
        message << "COLMAP step failed (" << name << ") with exit code " << code
                << ". See log: " << log_path.string();
        throw std::runtime_error(message.str());
    }
}

std::vector<std::string> sorted_model_dirs(const std::string& sparse_path) {
    std::vector<std::string> model_dirs;
    if (!fs::exists(sparse_path)) {
        return model_dirs;
    }
    for (const auto& entry : fs::directory_iterator(sparse_path)) {
        if (entry.is_directory()) {
            model_dirs.push_back(entry.path().filename().string());
        }
    }
    std::sort(model_dirs.begin(), model_dirs.end());
    return model_dirs;
}

py::dict run_external_cpu_sfm(
    const std::string& executable,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& sparse_path,
    const std::string& text_path,
    const std::vector<std::string>& image_names,
    const py::dict& options
) {
    fs::create_directories(fs::path(database_path).parent_path());
    fs::create_directories(sparse_path);
    fs::create_directories(text_path);
    const fs::path log_dir = fs::path(database_path).parent_path() / "command_logs";
    py::list command_logs;

    const std::string exe = shell_quote(executable);
    const std::string db = shell_quote(database_path);
    const std::string images = shell_quote(image_path);
    const std::string sparse = shell_quote(sparse_path);
    const std::string text = shell_quote(text_path);
    const std::string camera_model = option_value(options, "camera_model", "SIMPLE_RADIAL");
    const int max_features = option_int(options, "max_features", 8192);
    const int first_octave = option_int(options, "first_octave", 0);
    const int num_threads = option_int(options, "num_threads", -1);
    const int min_num_matches = option_int(options, "min_num_matches", 8);
    const int max_num_models = option_int(options, "max_num_models", 50);
    const int min_model_size = option_int(options, "min_model_size", static_cast<int>(image_names.size()));
    const int random_seed = option_int(options, "random_seed", 0);
    const int ba_global_max_refinements = option_int(options, "ba_global_max_refinements", 5);
    const bool multiple_models = option_bool(options, "multiple_models", true);
    const bool cross_check = option_bool(options, "cross_check", false);

    std::ostringstream feature;
    feature << exe << " feature_extractor"
            << " --random_seed " << random_seed
            << " --database_path " << db
            << " --image_path " << images
            << " --ImageReader.camera_model " << shell_quote(camera_model)
            << " --SiftExtraction.use_gpu 0"
            << " --SiftExtraction.max_num_features " << max_features
            << " --SiftExtraction.first_octave " << first_octave;
    if (num_threads > 0) {
        feature << " --SiftExtraction.num_threads " << num_threads;
    }
    run_command(feature.str(), "feature_extractor", log_dir, command_logs);

    std::ostringstream matcher;
    matcher << exe << " exhaustive_matcher"
            << " --random_seed " << random_seed
            << " --database_path " << db
            << " --SiftMatching.use_gpu 0"
            << " --SiftMatching.cross_check " << (cross_check ? 1 : 0)
            << " --TwoViewGeometry.min_num_inliers " << min_num_matches;
    if (num_threads > 0) {
        matcher << " --SiftMatching.num_threads " << num_threads;
    }
    run_command(matcher.str(), "exhaustive_matcher", log_dir, command_logs);

    std::ostringstream mapper;
    mapper << exe << " mapper"
           << " --random_seed " << random_seed
           << " --database_path " << db
           << " --image_path " << images
           << " --output_path " << sparse
           << " --Mapper.multiple_models " << (multiple_models ? 1 : 0)
           << " --Mapper.max_num_models " << max_num_models
           << " --Mapper.min_model_size " << min_model_size
           << " --Mapper.min_num_matches " << min_num_matches
           << " --Mapper.ba_global_max_refinements " << ba_global_max_refinements;
    if (options.contains("min_focal_length_ratio")) {
        mapper << " --Mapper.min_focal_length_ratio " << py::cast<double>(options[py::str("min_focal_length_ratio")]);
    }
    if (options.contains("max_focal_length_ratio")) {
        mapper << " --Mapper.max_focal_length_ratio " << py::cast<double>(options[py::str("max_focal_length_ratio")]);
    }
    if (options.contains("init_min_num_inliers")) {
        mapper << " --Mapper.init_min_num_inliers " << py::cast<int>(options[py::str("init_min_num_inliers")]);
    }
    if (options.contains("abs_pose_min_num_inliers")) {
        mapper << " --Mapper.abs_pose_min_num_inliers " << py::cast<int>(options[py::str("abs_pose_min_num_inliers")]);
    }
    if (options.contains("abs_pose_min_inlier_ratio")) {
        mapper << " --Mapper.abs_pose_min_inlier_ratio " << py::cast<double>(options[py::str("abs_pose_min_inlier_ratio")]);
    }
    if (options.contains("init_max_error")) {
        mapper << " --Mapper.init_max_error " << py::cast<double>(options[py::str("init_max_error")]);
    }
    if (options.contains("abs_pose_max_error")) {
        mapper << " --Mapper.abs_pose_max_error " << py::cast<double>(options[py::str("abs_pose_max_error")]);
    }
    if (options.contains("filter_max_reproj_error")) {
        mapper << " --Mapper.filter_max_reproj_error " << py::cast<double>(options[py::str("filter_max_reproj_error")]);
    }
    if (options.contains("abs_pose_refine_focal_length")) {
        mapper << " --Mapper.abs_pose_refine_focal_length " << (py::cast<bool>(options[py::str("abs_pose_refine_focal_length")]) ? 1 : 0);
    }
    if (options.contains("abs_pose_refine_extra_params")) {
        mapper << " --Mapper.abs_pose_refine_extra_params " << (py::cast<bool>(options[py::str("abs_pose_refine_extra_params")]) ? 1 : 0);
    }
    if (num_threads > 0) {
        mapper << " --Mapper.num_threads " << num_threads;
    }
    run_command(mapper.str(), "mapper", log_dir, command_logs);

    std::vector<std::string> model_dirs = sorted_model_dirs(sparse_path);
    for (const std::string& model_dir : model_dirs) {
        const fs::path source = fs::path(sparse_path) / model_dir;
        const fs::path target = fs::path(text_path) / model_dir;
        fs::create_directories(target);
        std::ostringstream convert;
        convert << exe << " model_converter"
                << " --input_path " << shell_quote(source.string())
                << " --output_path " << shell_quote(target.string())
                << " --output_type TXT";
        run_command(convert.str(), "model_converter_" + model_dir, log_dir, command_logs);
    }

    py::dict result;
    result["backend"] = "external_executable";
    result["model_ids"] = model_dirs;
    result["database_path"] = database_path;
    result["sparse_path"] = sparse_path;
    result["text_path"] = text_path;
    result["command_logs"] = command_logs;
    return result;
}

#ifdef MDIC_WITH_EMBEDDED_COLMAP
void wait_for_colmap_thread(colmap::Thread* thread, const std::string& name) {
    if (thread == nullptr) {
        throw std::runtime_error("COLMAP did not create controller for step: " + name);
    }
    thread->Start();
    thread->Wait();
}

void write_text_models(const colmap::ReconstructionManager& manager, const fs::path& text_path) {
    fs::create_directories(text_path);
    for (size_t idx = 0; idx < manager.Size(); ++idx) {
        const fs::path target = text_path / std::to_string(idx);
        fs::create_directories(target);
        manager.Get(idx)->WriteText(target);
    }
}

py::dict run_embedded_cpu_sfm(
    const std::string& database_path,
    const std::string& image_path,
    const std::string& sparse_path,
    const std::string& text_path,
    const std::vector<std::string>& image_names,
    const py::dict& options
) {
    fs::create_directories(fs::path(database_path).parent_path());
    fs::create_directories(sparse_path);
    fs::create_directories(text_path);

    const int num_threads = option_int(options, "num_threads", -1);
    const int random_seed = option_int(options, "random_seed", 0);
    const int min_num_matches = option_int(options, "min_num_matches", 8);

    colmap::ImageReaderOptions reader_options;
    reader_options.image_path = image_path;
    reader_options.image_names = image_names;
    reader_options.camera_model = option_value(options, "camera_model", "SIMPLE_RADIAL");
    reader_options.as_rgb = false;

    colmap::FeatureExtractionOptions extraction_options(colmap::FeatureExtractorType::SIFT);
    extraction_options.use_gpu = false;
    extraction_options.gpu_index = "-1";
    extraction_options.num_threads = num_threads;
    extraction_options.sift->max_num_features = option_int(options, "max_features", 8192);
    extraction_options.sift->first_octave = option_int(options, "first_octave", 0);
    if (options.contains("max_image_size")) {
        extraction_options.max_image_size = py::cast<int>(options[py::str("max_image_size")]);
    }

    colmap::FeatureMatchingOptions matching_options(colmap::FeatureMatcherType::SIFT_BRUTEFORCE);
    matching_options.use_gpu = false;
    matching_options.gpu_index = "-1";
    matching_options.num_threads = num_threads;
    matching_options.sift->cross_check = option_bool(options, "cross_check", false);

    colmap::TwoViewGeometryOptions geometry_options;
    geometry_options.min_num_inliers = min_num_matches;
    geometry_options.ransac_options.random_seed = random_seed;

    py::list steps;
    auto feature_extractor = colmap::CreateFeatureExtractorController(
        database_path, reader_options, extraction_options);
    wait_for_colmap_thread(feature_extractor.get(), "feature_extractor");
    steps.append(step_record("feature_extractor", "embedded_colmap"));

    const std::string matcher_name = option_value(options, "matcher", "exhaustive");
    std::unique_ptr<colmap::Thread> matcher;
    std::string matcher_step_name = "exhaustive_matcher";
    if (matcher_name == "ring" || matcher_name == "adjacent" || matcher_name == "imported") {
        const int matching_window = option_int(options, "matching_window", 2);
        const bool wrap_matching = option_bool(options, "wrap_matching", true);
        const auto pairs = ring_image_pairs(image_names, matching_window, wrap_matching);
        const fs::path pair_path = write_image_pair_list(database_path, pairs);
        colmap::ImportedPairingOptions pairing_options;
        pairing_options.match_list_path = pair_path;
        matcher = colmap::CreateImagePairsFeatureMatcher(
            pairing_options, matching_options, geometry_options, database_path);
        matcher_step_name = "ring_pair_matcher";
    } else if (matcher_name == "sequential") {
        colmap::SequentialPairingOptions pairing_options;
        pairing_options.overlap = option_int(options, "sequential_overlap", option_int(options, "matching_window", 2));
        pairing_options.quadratic_overlap = option_bool(options, "sequential_quadratic_overlap", false);
        pairing_options.loop_detection = option_bool(options, "loop_detection", false);
        matcher = colmap::CreateSequentialFeatureMatcher(
            pairing_options, matching_options, geometry_options, database_path);
        matcher_step_name = "sequential_matcher";
    } else {
        colmap::ExhaustivePairingOptions pairing_options;
        matcher = colmap::CreateExhaustiveFeatureMatcher(
            pairing_options, matching_options, geometry_options, database_path);
    }
    wait_for_colmap_thread(matcher.get(), matcher_step_name);
    steps.append(step_record(matcher_step_name, "embedded_colmap"));

    auto mapper_options = std::make_shared<colmap::IncrementalPipelineOptions>();
    mapper_options->image_path = image_path;
    mapper_options->image_names = image_names;
    mapper_options->multiple_models = option_bool(options, "multiple_models", true);
    mapper_options->max_num_models = option_int(options, "max_num_models", 50);
    mapper_options->min_model_size = std::min(
        option_int(options, "min_model_size", static_cast<int>(image_names.size())),
        static_cast<int>(image_names.size()));
    mapper_options->min_num_matches = min_num_matches;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    mapper_options->extract_colors = false;
    mapper_options->ba_use_gpu = false;
    mapper_options->ba_global_max_refinements = option_int(options, "ba_global_max_refinements", 5);
    mapper_options->min_focal_length_ratio = option_double(options, "min_focal_length_ratio", mapper_options->min_focal_length_ratio);
    mapper_options->max_focal_length_ratio = option_double(options, "max_focal_length_ratio", mapper_options->max_focal_length_ratio);
    mapper_options->mapper.init_min_num_inliers = option_int(options, "init_min_num_inliers", mapper_options->mapper.init_min_num_inliers);
    mapper_options->mapper.abs_pose_min_num_inliers = option_int(options, "abs_pose_min_num_inliers", mapper_options->mapper.abs_pose_min_num_inliers);
    mapper_options->mapper.abs_pose_min_inlier_ratio = option_double(options, "abs_pose_min_inlier_ratio", mapper_options->mapper.abs_pose_min_inlier_ratio);
    mapper_options->mapper.init_max_error = option_double(options, "init_max_error", mapper_options->mapper.init_max_error);
    mapper_options->mapper.abs_pose_max_error = option_double(options, "abs_pose_max_error", mapper_options->mapper.abs_pose_max_error);
    mapper_options->mapper.filter_max_reproj_error = option_double(options, "filter_max_reproj_error", mapper_options->mapper.filter_max_reproj_error);
    mapper_options->mapper.abs_pose_refine_focal_length = option_bool(
        options, "abs_pose_refine_focal_length", mapper_options->mapper.abs_pose_refine_focal_length);
    mapper_options->mapper.abs_pose_refine_extra_params = option_bool(
        options, "abs_pose_refine_extra_params", mapper_options->mapper.abs_pose_refine_extra_params);
    mapper_options->triangulation.random_seed = random_seed;

    auto database = colmap::Database::Open(database_path);
    auto reconstruction_manager = std::make_shared<colmap::ReconstructionManager>();
    colmap::IncrementalPipeline mapper(mapper_options, std::move(database), reconstruction_manager);
    mapper.Run();
    if (reconstruction_manager->Size() == 0) {
        throw std::runtime_error("Embedded COLMAP failed to create any sparse model.");
    }
    reconstruction_manager->Write(sparse_path);
    write_text_models(*reconstruction_manager, text_path);
    steps.append(step_record("incremental_mapping", "embedded_colmap"));
    steps.append(step_record("write_text_models", "embedded_colmap"));

    py::list model_ids;
    for (size_t idx = 0; idx < reconstruction_manager->Size(); ++idx) {
        model_ids.append(std::to_string(idx));
    }

    py::dict result;
    result["backend"] = "embedded_colmap";
    result["model_ids"] = model_ids;
    result["database_path"] = database_path;
    result["sparse_path"] = sparse_path;
    result["text_path"] = text_path;
    result["steps"] = steps;
    result["command_logs"] = py::list();
    return result;
}
#endif

py::dict run_cpu_sfm(
    const std::string& executable,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& sparse_path,
    const std::string& text_path,
    const std::vector<std::string>& image_names,
    const py::dict& options
) {
#ifdef MDIC_WITH_EMBEDDED_COLMAP
    (void)executable;
    return run_embedded_cpu_sfm(database_path, image_path, sparse_path, text_path, image_names, options);
#else
    return run_external_cpu_sfm(executable, database_path, image_path, sparse_path, text_path, image_names, options);
#endif
}

bool has_embedded_colmap() {
#ifdef MDIC_WITH_EMBEDDED_COLMAP
    return true;
#else
    return false;
#endif
}

py::dict capabilities() {
    py::dict result;
    result["embedded_colmap"] = has_embedded_colmap();
    result["cpu_only"] = true;
    result["feature_extraction"] = "sift";
    result["matching"] = "ring/imported, sequential, exhaustive";
    result["mapping"] = "incremental";
    result["exports"] = py::make_tuple("binary_colmap", "text_colmap");
    result["excluded"] = py::make_tuple("cuda", "gpu_sift", "gui", "dense_mvs", "meshing", "pycolmap_api");
    return result;
}

}  // namespace

PYBIND11_MODULE(native_colmap, m) {
    m.doc() = "CPU-only COLMAP SfM adapter used by Multi-DIC without depending on pycolmap.";
    m.def("has_embedded_colmap", &has_embedded_colmap);
    m.def("capabilities", &capabilities);
    m.def(
        "run_cpu_sfm",
        &run_cpu_sfm,
        py::arg("executable"),
        py::arg("database_path"),
        py::arg("image_path"),
        py::arg("sparse_path"),
        py::arg("text_path"),
        py::arg("image_names"),
        py::arg("options")
    );
}
