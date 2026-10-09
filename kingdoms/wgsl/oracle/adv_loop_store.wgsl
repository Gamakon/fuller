// A conditional store inside a loop with the same expression text before
// and after it: the two reads are two versions (the join's phi), and the
// read after the loop a third.
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> out2: array<f32>;
@group(0) @binding(3) var<storage, read_write> out3: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = s * s + xs[i];
        if (xs[i] > 0.0) { s = s + xs[i]; }
        out2[i] = s * s + xs[i];
    }
    out3[0] = s * s;
}
