#version 450

// Empty fragment shader for the depth-only shadow pass: the pipeline has no color attachment,
// rasterization depth (from gl_Position) is all the pass needs, and no output is declared.
void main() {
}
