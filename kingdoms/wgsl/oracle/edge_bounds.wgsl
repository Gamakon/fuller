// Indexing past the end under the bounds policy wgpu 30 applies on Metal,
// naga's Restrict (metal::min(index, len - 1)): a load past the end reads the
// LAST element, a store past the end writes the last element. xs has 3 lanes,
// out has 4; the loop runs 8. By hand: out[0] = xs[1] + 1 = 21, out[1] = xs[2] + 1
// = 31, out[2] = xs[3 -> 2] + 1 = 31, out[3] = xs[4 -> 2] + 1 = 31, and the
// stores out[4..8] all land on out[3] (31 each); then out[3] = out[8 -> 3] + 2
// = 33. Result [21, 31, 31, 33].
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = xs[i + 1u] + 1.0;
    }
    out[3] = out[8] + 2.0;
}
