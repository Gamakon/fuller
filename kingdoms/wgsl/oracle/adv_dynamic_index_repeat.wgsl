// Dynamic indexing with repeated loads of a buffer that is stored between
// the repeats: the texts before and after the store are two versions and
// may not share; the repeat within one version (ys[j] * ys[j], twice
// before the store) may.
@group(0) @binding(0) var<storage, read> ks: array<u32>;
@group(0) @binding(1) var<storage, read_write> ys: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> out2: array<f32>;
@group(0) @binding(4) var<storage, read_write> out3: array<f32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = ks[i];
        out[i] = ys[j] * ys[j] + ys[j];
        out2[i] = ys[j] * ys[j] - 1.0;
        ys[j] = ys[j] * 2.0;
        out3[i] = ys[j] * ys[j] + ys[j];
    }
}
