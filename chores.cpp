module;

// The global-module-fragment include below is load-bearing, not stylistic: with -fno-exceptions
// and the vendored std module, a chores implementation unit that instantiates std::vector (this
// file does, in load_shader) sees TWO 'operator new(size_t, align_val_t)' declarations - module
// std's and the textual libc++ copy baked into utility.data_block.pcm (data_block is the one
// module that never imports std; it includes libc++ headers in its own GMF). The result is
// 'call to operator new is ambiguous' at allocate.h. Textually including glm here (the same
// trick vulkan/animation/animation_controller.cpp uses) makes clang merge the two copies, so
// the allocator instantiations resolve. Do not remove this include to "clean up".
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

module chores;

namespace chores {
    // Resolve the startup config in one step: merge config file + argv into the app settings,
    // then locate the shaders/ dir and pick the model file. Panics when a resource is missing.
    startup_config analyse_config(int argc, char** argv) {
        // 1. Resolve startup settings first: config file (config.toml by default, --config <path>
        //    to override) merged with positional argv overrides. argv[1] = model, argv[2] = grid
        //    side (numeric) or demo, argv[3] = demo.
        app_config::app_settings const settings = app_config::resolve_from_argv(argc, argv);
        if (!settings.config_file.empty()) {
            utility::log("app_config: loaded startup settings from '{}'", settings.config_file);
        }

        startup_config config = {settings, {}, {}};

        // 2. Shaders directory (holds GLSL sources and compiled SPIR-V): explicit config path when
        //    given, otherwise walk up from the working directory to find shaders/.
        if (!settings.paths.shaders_dir.empty()) {
            config.shaders_dir = settings.paths.shaders_dir;
            if (!std::filesystem::is_directory(config.shaders_dir)) {
                utility::panic(std::source_location::current(), "cannot find configured shaders_dir '{}'.", settings.paths.shaders_dir);
            }
        } else if (std::optional<std::filesystem::path> const located = locate_shaders_dir()) {
            config.shaders_dir = *located;
        } else {
            utility::panic("cannot find shaders/ directory. run the program from the project root, pass shaders_dir in config.toml, or use a cmake-build-* directory.");
        }

        // 3. Pick the model file: settings.model when configured/argv-given; else the default model
        //    under settings.paths.model_dir (or the auto-located gltf_model/).
        if (!settings.model.empty()) {
            config.model_path = settings.model;
        } else if (!settings.paths.model_dir.empty()) {
            config.model_path = (std::filesystem::path(settings.paths.model_dir) / "DamagedHelmet.gltf").string();
            if (!std::filesystem::is_regular_file(config.model_path)) {
                utility::panic(std::source_location::current(), "cannot find model '{}' under configured model_dir '{}'.", "DamagedHelmet.gltf", settings.paths.model_dir);
            }
        } else if (std::optional<std::filesystem::path> const located = locate_model_file()) {
            config.model_path = located->string();
        } else {
            utility::panic("cannot find gltf_model/DamagedHelmet.gltf. run the program from the project root or pass a model path as argv[1]");
        }

        return config;
    }

    // Read a single shader SPIR-V file and print info; panic on failure
    void load_shader(std::filesystem::path const& dir, std::string_view const file_name, std::vector<unsigned char>& out) {
        std::filesystem::path const path = dir / file_name;
        std::optional<std::vector<unsigned char>> const data = utility::read_binary_to_vector(path);
        if (!data) {
            utility::panic(std::source_location::current(), "cannot open shader file '{}'", path.string());
        }
        out = *data;
        utility::log("loaded shader: {} ({} bytes)", path.string(), out.size());
    }

    // Walk up from the working directory to find the shaders/ directory,
    // so it works when run from the project root or a cmake-build-* directory
    std::optional<std::filesystem::path> locate_shaders_dir() {
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

    // Walk up from the working directory to find the default model under gltf_model/
    std::optional<std::filesystem::path> locate_model_file() {
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

    // Load a vertex/fragment SPIR-V pair and create the pipeline via runtime; panic on failure
    void load_and_create_pipeline(vulkan::runtime& runtime,
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

    // Create the demo pipelines up front: the standard PBR pipeline (used by the imported scene)
    // plus the skybox background (fullscreen environment pass) and the directional shadow pass
    // (depth-only). The legacy triangle demo pipeline is deliberately not created here - nothing
    // draws it anymore.
    void setup_pipeline(vulkan::runtime& runtime, std::filesystem::path const& shaders_dir) {
        // Standard PBR pipeline: the imported scene's primitives bind to it
        load_and_create_pipeline(runtime, shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv");

        {
            // Skybox background pipeline (fullscreen environment pass): drawn first every frame,
            // behind the scene. Panic on failure - the frame needs a background to clear to.
            std::vector<unsigned char> vertex_code;
            std::vector<unsigned char> fragment_code;
            load_shader(shaders_dir, "skybox.vert.spv", vertex_code);
            load_shader(shaders_dir, "skybox.frag.spv", fragment_code);
            auto const skybox_result = runtime.make_skybox_pipeline(vertex_code, fragment_code);
            if (!skybox_result) {
                utility::panic(std::source_location::current(), "failed to create skybox pipeline: {}", skybox_result.error());
            }
            utility::log("SUCCESS: skybox pipeline created (fullscreen environment background)");
        }

        {
            // Shadow pass pipeline (depth-only): renders the scene from the light into the shadow
            // map. Created once; enable_shadows() activates the pass after the scene import.
            // Failure is not fatal - the scene simply renders without shadows.
            std::vector<unsigned char> vertex_code;
            std::vector<unsigned char> fragment_code;
            load_shader(shaders_dir, "shadow.vert.spv", vertex_code);
            load_shader(shaders_dir, "shadow.frag.spv", fragment_code);
            auto const shadow_result = runtime.make_shadow_pipeline(vertex_code, fragment_code);
            if (!shadow_result) { // NOLINT(bugprone-branch-clone): CLion FP - the branches log different messages
                utility::log("shadow pipeline disabled: {}", shadow_result.error());
            } else {
                utility::log("SUCCESS: shadow pipeline created (directional shadow map pass)");
            }
        }
    }

    // Optional instancing stress: grid_side > 1 (config or argv) draws the first imported
    // primitive as a grid_side x grid_side grid in ONE instanced draw call (an
    // instanced_draw_primitive appended to the scene tree - the frame loop is untouched)
    void add_instancing_grid(vulkan::runtime& runtime, int const grid_side, float const scene_radius) {
        if (grid_side <= 1) {
            return;
        }
        std::vector<vulkan::primitive const*> const pbr_primitives = runtime.get_primitives("pbr");
        if (pbr_primitives.empty()) {
            return;
        }
        vulkan::primitive const& source = *pbr_primitives[0];
        std::vector<glm::mat4> transforms;
        transforms.reserve(static_cast<size_t>(grid_side) * grid_side);
        float const spacing = 2.5f * scene_radius; // keep instances apart: measure draw scaling, not overdraw
        for (int i = 0; i < grid_side; ++i) {
            for (int j = 0; j < grid_side; ++j) {
                float const dx = (static_cast<float>(i) - static_cast<float>(grid_side - 1) * 0.5f) * spacing;
                float const dz = (static_cast<float>(j) - static_cast<float>(grid_side - 1) * 0.5f) * spacing;
                transforms.push_back(glm::translate(glm::mat4(1.0f), glm::vec3(dx, 0.0f, dz)) * source.push.model);
            }
        }
        runtime.make_instanced_primitive(source, transforms);
        utility::log("instancing stress: {} x {} grid ({} instances, 1 draw call)", grid_side, grid_side, transforms.size());
    }
} // namespace chores
