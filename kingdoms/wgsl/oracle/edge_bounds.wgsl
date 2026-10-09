// Indexing past the end under wgpu's ReadZeroSkipWrite policy on this
// platform: a load past the end reads zero, a store past the end is
// skipped. xs has 3 lanes, out has 4; the loop runs 8.
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = xs[i + 1u] + 1.0;
    }
    out[3] = out[8] + 2.0;
}
