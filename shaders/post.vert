#version 450

// Post-process fullscreen pass: no vertex buffer, the triangle is synthesized from
// gl_VertexIndex (covers the whole screen with 3 vertices). UV follows the framebuffer
// orientation, so v_uv = (0,0) is the top-left texel of the HDR scene target.
layout(location = 0) out vec2 v_uv;

void main() {
    vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
