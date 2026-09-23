// shaders/post.vert - the SLANG source of the fullscreen-triangle vertex stage.
//
// PRODUCES `post.vert.spv`, the same name the GLSL version produced, so the host is untouched. Every
// fullscreen pass uses it (the bloom/composite chain, TAA, FXAA, deferred lighting and the G-buffer debug
// view), so this one file is the vertex stage of five pipelines - which is why it is ported early: it is
// also the cheapest place to prove a VERTEX stage in Slang, before pbr.vert and shadow.vert need one.
//
// No vertex buffer: the triangle is synthesized from the vertex index, which covers the whole screen with
// three vertices and avoids the diagonal seam a two-triangle quad can show with some filters. The UV
// follows the framebuffer orientation, so v_uv = (0,0) is the top-left texel of the source image.

struct post_vertex_output
{
    [[vk::location(0)]] float2 v_uv;
    float4 position : SV_Position; // gl_Position on the GLSL side; SV_Position is the same builtin
};

[shader("vertex")]
post_vertex_output main(uint vertex_index : SV_VertexID)
{
    // GLSL's gl_VertexIndex is SPIR-V's VertexIndex; SV_VertexID subtracts BaseVertex, which a
    // fullscreen draw never sets - the two are the same value here (Slang's own docs state the mapping).
    float2 pos = float2(float((vertex_index << 1u) & 2u), float(vertex_index & 2u));

    post_vertex_output out;
    out.v_uv = pos;
    out.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
