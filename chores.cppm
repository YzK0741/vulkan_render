module;

#include <glm/glm.hpp>

export module chores;

import std;
import utility;
import vulkan.runtime;

/**
 * @file chores.cppm
 * @defgroup chores Demo Bootstrap Chores
 * @brief main()'s startup helper functions, kept out of main.cpp so the entry point reads
 *        first: loading shader SPIR-V files, locating the shaders/ and default-model dirs by
 *        walking up from the working directory, and creating pipelines from a vertex/fragment
 *        SPIR-V pair through the runtime. Demo/application layer only - these helpers know
 *        about the runtime (vulkan.runtime) and the utility log/panic, never about glTF.
 * @note header-only module (implementations inline): a split implementation unit trips over
 *       the vendored std module's operator-new declarations under -fno-exceptions
 */
namespace chores {
    /**
     * @ingroup chores
     * @brief read one shader SPIR-V file into @p out; panic on failure (prints the path)
     */
    export void load_shader(std::filesystem::path const& dir, std::string_view const file_name, std::vector<unsigned char>& out) {
        std::filesystem::path const path = dir / file_name;
        std::optional<std::vector<unsigned char>> const data = utility::read_binary_to_vector(path);
        if (!data) {
            utility::panic(std::source_location::current(), "cannot open shader file '{}'", path.string());
        }
        out = *data;
        utility::log("loaded shader: {} ({} bytes)", path.string(), out.size());
    }

    /**
     * @ingroup chores
     * @brief walk up from the working directory to find the shaders/ directory (works from the
     *        project root or a cmake-build-* directory); nullopt when not found within 4 levels
     */
    export std::optional<std::filesystem::path> locate_shaders_dir() {
        std::filesystem::path current = std::filesystem::current_path();
        for (int depth = 0; depth < 4; ++depth) {
            std::filesystem::path candidate = current / "shaders";
            if (std::filesystem::is_directory(candidate)) {
                return candidate;
            }
            std::filesystem::path const parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    /**
     * @ingroup chores
     * @brief walk up from the working directory to find the default model under gltf_model/
     *        (gltf_model/DamagedHelmet.gltf); nullopt when not found within 4 levels
     */
    export std::optional<std::filesystem::path> locate_model_file() {
        std::filesystem::path current = std::filesystem::current_path();
        for (int depth = 0; depth < 4; ++depth) {
            std::filesystem::path candidate = current / "gltf_model" / "DamagedHelmet.gltf";
            if (std::filesystem::is_regular_file(candidate)) {
                return candidate;
            }
            std::filesystem::path const parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    /**
     * @ingroup chores
     * @brief load a vertex/fragment SPIR-V pair and create the pipeline via the runtime; panic
     *        on load or creation failure
     */
    export void load_and_create_pipeline(vulkan::runtime& runtime,
                                         std::filesystem::path const& shaders_dir,
                                         std::string_view const pipeline_name,
                                         std::string_view const vertex_file,
                                         std::string_view const fragment_file) {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        load_shader(shaders_dir, vertex_file, vertex_code);
        load_shader(shaders_dir, fragment_file, fragment_code);

        std::expected<void, std::string> const result = runtime.make_pipeline(pipeline_name, vertex_code, fragment_code);
        if (!result) {
            utility::panic(std::source_location::current(), "failed to create pipeline '{}': {}", pipeline_name, result.error());
        }
        utility::log("SUCCESS: pipeline '{}' created and cached in the runtime", pipeline_name);
    }
} // namespace chores
