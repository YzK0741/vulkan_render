#version 450

/**
 * @file shaders/skybox.vert
 * @brief Fullscreen-triangle skybox: emits a world-space view direction per screen corner.
 * @ingroup shaders
 *
 * No vertex buffer, 3 vertices. Each vertex emits the world-space view direction for its screen
 * corner; skybox.frag normalizes the interpolated direction and evaluates the analytic sky. The
 * pipeline is created by runtime::make_skybox_pipeline() and shares the scene set (the camera UBO
 * at binding 0 is all this pass needs).
 */
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} camera;

layout(location = 0) out vec3 v_dir;

/**
 * @brief emit the fullscreen triangle and the world-space view ray for its corner
 */
void main() {
    // Fullscreen triangle covering the viewport (no vertex input). Winding flipped vs. the naive
    // (id<<1)&2, id&2 pattern so the triangle is front-facing with the pipeline's BACK culling
    // (VK_FRONT_FACE_COUNTER_CLOCKWISE, framebuffer coords with y down).
    vec2 uv = vec2(float(gl_VertexIndex & 2), float((gl_VertexIndex << 1) & 2));
    vec2 ndc = uv * 2.0 - 1.0;

    // View-space ray direction: the camera translation does not matter for an infinite skybox
    vec4 eye = inverse(camera.proj) * vec4(ndc, 1.0, 1.0);
    // Rotate into world space; the fragment normalizes the interpolated direction
    v_dir = inverse(mat3(camera.view)) * eye.xyz;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
