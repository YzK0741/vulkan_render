#version 450

/**
 * @file shaders/outline.frag
 * @brief The inverted hull's fragment stage: the G-buffer's five targets, filled with the outline colour.
 * @ingroup shaders
 *
 * It writes the SAME five attachments gbuffer.frag writes, because the hull goes through the same pipeline
 * shape and the same lighting stage reads the result: an outline is not an overlay drawn on top of the
 * frame, it is geometry whose surface happens to be the line. That is also why the colour is written into
 * ALBEDO rather than into the scene colour - the model's own outline is then lit, shadowed and tonemapped
 * with everything else, which is what keeps it attached to the model.
 *
 * Two fields are chosen to make the hull behave like a flat line rather than like a surface:
 *   - metallic 0 and roughness 1: no specular lobe to speak of, so what the lighting returns is the albedo
 *     (the outline colour) modulated by the light that reaches it;
 *   - the material flags byte is 0, so the lighting stage cannot mistake the hull for an alphaMode
 *     MASK/BLEND surface and discard parts of it - a hull that is discarded in patches is a broken line.
 *
 * THE VELOCITY IS ZERO, and that is a known limit rather than an oversight: the hull carries no
 * previous-frame world matrix (see outline.vert), so TAA reprojects it as if it were static. On a moving
 * character the line therefore gets the camera's motion only, which shows up as a slight crawl along the
 * silhouette. docs/zzz_shading.md records it with the rest of what this slice does not do yet.
 */

layout(location = 0) in vec3 v_normal;

layout(location = 0) out vec4 out_albedo_metallic;  // rgb albedo, a metallic
layout(location = 1) out vec4 out_normal_roughness; // xyz world normal, w roughness
layout(location = 2) out vec4 out_material;         // r/g material id, b ao, a flags
layout(location = 3) out vec2 out_velocity;         // motion vector in UV space
layout(location = 4) out vec4 out_scene_color;      // the emissive term (the hull emits nothing)

#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : enable
#include "heap_slots.glsl"

// The camera block, declared in full for the reason outline.vert gives: the outline lane is the LAST
// member, so everything before it has to be declared for its offset to be right.
layout(descriptor_heap, descriptor_stride = heap_slot_stride) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float camera_padding;
    mat4 view_proj_unjittered;
    mat4 prev_view_proj;
    vec4 outline;
} camera[];

// The material push block, for the id only. Its layout must mirror material_push_constants
// (vulkan/primitive/primitive.cppm) up to the field this stage reads - and the fields it does NOT read
// still have to be declared up to that point (see outline.vert's note on `motion_base`).
layout(push_constant) uniform PushConstants {
    uint material_index;
    uint flags;
    uint skin_base;
    uint morph_base;
    uint morph_targets;
    uint morph_vertices;
    uint instance_base;
    uint motion_base;
    mat4 model;
    uint frame_slot;
    uint image_index;
} push;

#define heap_frame_slot (push.frame_slot)
#define heap_image_index (push.image_index)

// The material table, for this draw's record: the COLOUR is per material in MMD (usually black, the hair's
// is a dark green here - see docs/zzz_shading.md), so the frame's [render] outline_color is only the
// fallback for a material that authored none. Declared in full - a storage buffer's array stride is the
// struct's own size, so a copy that stops early indexes the table at the wrong pitch.
struct Material {
    uvec4 tex_indices;
    uint emissive_index;
    float alpha_cutoff;
    float occlusion_strength;
    uint sphere_index; // MMD sphere map: the texture it was combined from (0 = none); flags bits 7-8 hold the mode
    vec4 base_color_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    uint flags; // bit6: the model authored MMD edge data
    vec4 npr_edge;
};
layout(descriptor_heap, descriptor_stride = heap_slot_stride) readonly buffer Materials { Material materials[]; } heap_material_tables[];

void main() {
    const Material material = heap_material_tables[heap_slots_materials].materials[push.material_index];
    const vec3 line = (material.flags & 1024u) != 0u ? material.npr_edge.rgb : camera[heap_camera_slot].outline.rgb;
    out_albedo_metallic = vec4(line, 0.0);
    // A degenerate interpolated normal would normalize to a NaN, which the lighting stage would then read
    // as a normal - so it falls back to a facing direction instead.
    const float normal_length = length(v_normal);
    out_normal_roughness = vec4(normal_length > 1e-8 ? v_normal / normal_length : vec3(0.0, 0.0, 1.0), 1.0);

    const uint id = push.material_index;
    out_material = vec4(float(id & 0xFFu) / 255.0, float((id >> 8u) & 0xFFu) / 255.0, 1.0, 0.0);

    out_velocity = vec2(0.0);
    out_scene_color = vec4(0.0);
}
