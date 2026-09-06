module;

#include <glm/glm.hpp>

export module chores;

export import std;
import app_config;
import utility;
import vulkan.runtime;

/**
 * @file chores.cppm
 * @defgroup chores Demo Bootstrap Chores
 * @brief main()'s startup helper functions, kept out of main.cpp so the entry point reads
 *        first: resolving the startup config (config file + argv merge, shaders/ and
 *        default-model dirs), creating the demo pipelines, loading shader SPIR-V files, and
 *        the optional instancing stress grid. Demo/application layer only - these helpers
 *        know about app_config (settings), vulkan.runtime and the utility log/panic, never
 *        about glTF (the loader's own diagnostics live in gltf_loader).
 * @note interface only - the implementations live in chores.cpp (the module follows the
 *       export import std pattern of the other split modules)
 */
namespace chores {
    /**
     * @ingroup chores
     * @brief the resolved startup environment: merged settings + located shaders dir + model path
     * @note everything main needs before it touches the runtime; produced by analyse_config()
     */
    export struct startup_config {
        app_config::app_settings settings = {}; // merged config-file + argv settings
        std::filesystem::path shaders_dir = {}; // shader SPIR-V directory (configured or located)
        std::string model_path = {};            // glTF file to load (configured, model_dir, or located)
    };

    /**
     * @ingroup chores
     * @brief resolve the startup config in one step: merge the config file + argv into the
     *        app settings, then locate the shaders/ directory and pick the model file
     *        (configured path, model_dir default, or auto-located gltf_model/). Panics when a
     *        configured or located resource is missing.
     * @param argc argv argument count (as received by main)
     * @param argv argv argument vector (as received by main)
     */
    export startup_config analyse_config(int argc, char** argv);

    /**
     * @ingroup chores
     * @brief read one shader SPIR-V file into @p out; panic on failure (prints the path)
     */
    export void load_shader(std::filesystem::path const& dir, std::string_view const file_name, std::vector<unsigned char>& out);

    /**
     * @ingroup chores
     * @brief walk up from the working directory to find the shaders/ directory (works from the
     *        project root or a cmake-build-* directory); nullopt when not found within 4 levels
     */
    export std::optional<std::filesystem::path> locate_shaders_dir();

    /**
     * @ingroup chores
     * @brief walk up from the working directory to find the default model under gltf_model/
     *        (gltf_model/DamagedHelmet.gltf); nullopt when not found within 4 levels
     */
    export std::optional<std::filesystem::path> locate_model_file();

    /**
     * @ingroup chores
     * @brief load a vertex/fragment SPIR-V pair and create the pipeline via the runtime; panic
     *        on load or creation failure
     */
    export void load_and_create_pipeline(vulkan::runtime& runtime,
                                         std::filesystem::path const& shaders_dir,
                                         std::string_view const pipeline_name,
                                         std::string_view const vertex_file,
                                         std::string_view const fragment_file);

    /**
     * @ingroup chores
     * @brief create the demo pipelines up front: the standard PBR pipeline plus the skybox
     *        background (fullscreen environment pass) and the directional shadow-map pass
     *        (depth-only). Panics when PBR/skybox creation fails; a shadow failure only logs
     *        "shadow pipeline disabled" (the scene still renders without shadows).
     * @note the old triangle demo pipeline is deliberately NOT created here: nothing draws it
     *       anymore (the skybox background and the imported scene cover the screen)
     */
    export void setup_pipeline(vulkan::runtime& runtime, std::filesystem::path const& shaders_dir);

    /**
     * @ingroup chores
     * @brief optional instancing stress: when @p grid_side > 1 (config or argv), draw the first
     *        imported "pbr" primitive as a grid_side x grid_side grid in ONE instanced draw call
     *        (an instanced_draw_primitive appended to the scene tree — the frame loop is
     *        untouched). No-op when grid_side <= 1 or the scene has no pbr primitive.
     * @param runtime the initialized runtime holding the imported scene
     * @param grid_side grid side length from settings.grid_side (> 1 enables the grid)
     * @param scene_radius radius of the imported scene (grid spacing = 2.5 x radius, so
     *        instances stay apart: the demo measures draw scaling, not overdraw)
     */
    export void add_instancing_grid(vulkan::runtime& runtime, int grid_side, float scene_radius);
} // namespace chores
