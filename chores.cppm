module;

#include <glm/glm.hpp>

export module chores;

export import std;
import app_config;
import utility;
import vulkan.animation; // setup_gui builds the animation playback controls
import vulkan.runtime;

/**
 * @file chores.cppm
 * @defgroup chores Demo Bootstrap Chores
 * @brief main()'s startup helper functions, kept out of main.cpp so the entry point reads
 *        first: resolving the startup config (config file + argv merge, shaders/ and
 *        default-model dirs), creating the demo pipelines, loading shader SPIR-V files, the
 *        optional instancing stress grid, and assembling the demo's Dear ImGui debug overlay
 *        (setup_gui). Demo/application layer only - these helpers know about app_config
 *        (settings), vulkan.runtime and the utility log/panic, never about glTF (the
 *        loader's own diagnostics live in gltf_loader).
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

    /**
     * @ingroup chores
     * @brief live state the debug-gui widgets bind to. Owned by main (the frame loop keeps the
     *        fps text and the animation mirrors in sync each frame); setup_gui() wires the
     *        widgets to these fields.
     * @note defaults mirror the historic demo values; main overrides the ones that come from
     *       config ([render] skybox/shadow toggles) or runtime state (animation playing)
     */
    export struct gui_bindings {
        double fps = 0.0;                  // fps text (updated once per second when use_gui)
        bool cull_enabled = true;          // frustum-culling checkbox (write-through to the runtime)
        bool skybox_enabled = true;        // skybox checkbox (initial: settings.render.skybox)
        bool shadow_enabled = true;        // shadow checkbox (initial: settings.render.shadow)
        float shadow_bias_constant = 0.0f; // shadow depth-bias sliders (constant factor)
        float shadow_bias_slope = 1.5f;    // shadow depth-bias sliders (slope factor)
        float anim_time = 0.0f;            // animation time slider (mirror of the controller clock)
        bool anim_playing = true;          // play/pause checkbox (mirror of controller state)
        int anim_index = 0;                // animation combo selection (0 = the auto-played one)
        int current_camera = 0;            // camera combo selection (0 = orbit, 1..N = authored)
    };

    /**
     * @ingroup chores
     * @brief build the demo's Dear ImGui debug overlay when @p use_gui: enable it on the
     *        runtime and assemble the "vulkan_render debug" panel — fps label, frustum-culling
     *        / skybox / shadow toggles, the camera-target drag, animation playback controls
     *        (label + play + time scrubber + animation dropdown when the controller has an
     *        active animation), the camera selector (when @p camera_names is non-empty) and
     *        the shadow depth-bias sliders.
     * @param runtime the initialized runtime (enable_debug_gui() is called here)
     * @param use_gui whether the overlay is wanted ([gui] show or the "gui" demo); no-op when false
     * @param settings startup settings: [gui] panel size + [render] initial skybox/shadow states
     * @param bindings live widget state (see gui_bindings); the frame loop updates fps and the
     *        animation mirrors each frame
     * @param animation the animation controller the playback widgets drive (may be idle)
     * @param camera_names display names of the scene's authored cameras (no "orbit" entry is
     *        added here — setup_gui prepends it); empty disables the camera selector
     * @param on_camera_selected called with the combo index (0 = orbit, i = camera_names[i-1])
     *        when the user picks a camera; empty when the selector is not shown
     * @note the overlay's glTF-side data (authored camera list, orbit seeding) stays in main:
     *       chores only ever sees display names and a callback, never glTF types
     */
    export void setup_gui(vulkan::runtime& runtime,
                          bool use_gui,
                          app_config::app_settings const& settings,
                          gui_bindings& bindings,
                          vulkan::animation::controller& animation,
                          std::vector<std::string> const& camera_names,
                          std::function<void(int)> const& on_camera_selected);

    /**
     * @ingroup chores
     * @brief assemble an animation::backend for @p runtime: the host surface an
     *        animation::controller drives (scene + per-slot buffer callbacks + the task pool),
     *        wired to the runtime's own scene, frame-slot buffers and run_tasks. Pass it to
     *        controller::init(); the controller itself never depends on vulkan::runtime.
     */
    export vulkan::animation::backend make_animation_backend(vulkan::runtime& runtime);
} // namespace chores
