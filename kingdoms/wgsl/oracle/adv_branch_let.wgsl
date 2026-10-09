// An expression valid only within a branch: the let is bound in the arm
// and the repeated text reads it there; after the join the local carries
// the value out.
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> out2: array<f32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 8u; i = i + 1u) {
        var r = -1.0;
        if (i < n) {
            let v = xs[i];
            r = v * v + 1.0;
            out[i] = v * v + 1.0;
        } else {
            out[i] = -2.0;
        }
        out2[i] = r;
    }
}
