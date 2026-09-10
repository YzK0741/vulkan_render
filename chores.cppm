module;

#include <glm/glm.hpp>

export module chores;

export import vstd;
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
        int render_mode = 0;               // render-mode combo (0 = pbr, 1 = unlit); main applies it
                                           // between frames via set_default_pipeline
        int brdf_model = 0;                // brdf-model combo (0 = GGX+joint, 1 = GGX+height-corr,
                                           // 2 = Beckmann, 3 = Blinn-Phong); write-through to runtime
        int diffuse_model = 0;             // diffuse combo (0 = Lambert, 1 = Oren-Nayar)
        float exposure = 1.0f;             // linear exposure slider (runtime::set_exposure)
        float bloom_intensity = 0.8f;      // bloom blend weight slider (runtime::set_bloom; 0 = off)
        int toon_bands_index = 0;          // cel-shading combo: index into toon_band_counts (0 = plain PBR)
        float toon_softness = 0.15f;       // cel-shading band edge softness slider (smaller = harder edges)
        float bloom_threshold = 0.35f;     // bloom bright-pass threshold (visible range 0..0.75)
        // FXAA (runtime::set_fxaa): checkbox + the two shader knobs. The checkbox is mirrored by
        // main into the runtime every frame like the other post-process values.
        bool fxaa_enabled = false;         // FXAA on/off (initial: settings.render.fxaa)
        float fxaa_subpixel = 0.75f;       // sub-pixel term strength (0 = pure directional blend)
        float fxaa_edge_threshold = 0.166f; // relative luma contrast below which a pixel is "flat"
        // G-buffer debug view (runtime::set_gbuffer_debug / set_gbuffer_channel): the deferred
        // path's stored surface, one channel at a time. Mirrored into the runtime every frame like
        // the FXAA state, so these fields carry the config's initial values.
        bool gbuffer_debug = false; // draw the G-buffer + its debug view instead of the shaded scene
        int gbuffer_channel = 1;    // 0 albedo, 1 normal, 2 roughness, 3 metallic, 4 ao, 5 id, 6 depth, 7 flags
        // deferred lighting (runtime::set_deferred): shade the opaque scene from the G-buffer instead
        // of forward. Precedence is deliberate: the debug view wins when both are on, so the checkbox
        // always shows what is actually stored.
        bool deferred_enabled = false;
        // TAA (runtime::set_taa): the deferred path's anti-aliasing. Mirrored into the runtime every
        // frame like the other render toggles; the blend weights are the two shader knobs.
        bool taa_enabled = false;
        float taa_blend_static = 0.9f; // history weight for a static pixel (0.9 = 10% of the new frame)
        float taa_blend_min = 0.5f;    // history weight floor under motion (lower = less ghosting)
        // demo punctual lights (count matches vulkan::max_punctual_lights): the gui rows below
        // edit these fields live (no per-widget callbacks), and main() pushes the enabled set to
        // the runtime once per frame via chores::apply_point_lights(). Each slot is a point light
        // or - with `spot` set - a cone light (direction + inner/outer half-angles in degrees;
        // apply_point_lights clamps inner <= outer < 90 and converts to the shader's cosines).
        // Plain C arrays keep this interface glm-free.
        struct light_slot {
            bool enabled = false;
            bool spot = false;                        // false = point light (omni), true = spot cone
            float position[3] = {0.0f, 0.0f, 0.0f};   // world position
            float direction[3] = {0.0f, -1.0f, 0.0f}; // spot axis (spot only; normalized when pushed)
            float color[3] = {1.0f, 1.0f, 1.0f};      // linear color
            float intensity = 1.0f;
            float range = 10.0f;          // 0 = infinite falloff
            float inner_cone_deg = 20.0f; // spot: soft inner half-angle (degrees)
            float outer_cone_deg = 30.0f; // spot: hard cutoff half-angle (degrees)
        };
        light_slot point_lights[4] = {};
    };

    /**
     * @ingroup chores
     * @brief build the demo's Dear ImGui debug overlay when @p use_gui: enable it on the
     *        runtime and assemble the "vulkan_render debug" panel — fps label, frustum-culling
     *        / skybox / shadow toggles, the camera-target drag, animation playback controls
     *        (label + play + time scrubber + animation dropdown when the controller has an
     *        active animation), the camera selector (when @p camera_names is non-empty), the
     *        shadow depth-bias sliders and the punctual light rows (per-slot enable / position /
     *        color / intensity / range plus spot cone editing: spot toggle, direction, inner and
     *        outer half-angles).
     * @param runtime the initialized runtime (enable_debug_gui() is called here)
     * @param use_gui whether the overlay is wanted ([gui] show); no-op when false
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
     * @brief push the enabled punctual-light slots of @p bindings into the runtime's light UBO.
     *        Called once per frame from main (while the gui is active): the gui widgets edit
     *        bindings.point_lights live, so a drag/toggle becomes visible next frame without
     *        per-widget callbacks. Each slot is pushed as a point light, or as a spot light when
     *        its `spot` flag is set (direction + clamped inner/outer cone angles). Cheap no-op
     *        when nothing is enabled.
     */
    export void apply_point_lights(vulkan::runtime& runtime, gui_bindings const& bindings);

    /**
     * @ingroup chores
     * @brief assemble an animation::backend for @p runtime: the host surface an
     *        animation::controller drives (scene + per-slot buffer callbacks + the task pool),
     *        wired to the runtime's own scene, frame-slot buffers and run_tasks. Pass it to
     *        controller::init(); the controller itself never depends on vulkan::runtime.
     */
    export vulkan::animation::backend make_animation_backend(vulkan::runtime& runtime);
} // namespace chores
