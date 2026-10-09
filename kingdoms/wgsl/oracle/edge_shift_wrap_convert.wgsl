// Shifts mask their amount to the low five bits, integer add and multiply
// wrap, and a float converts to an integer toward zero with saturation.
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> n: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read_write> w: array<u32>;
@group(0) @binding(4) var<storage, read> f: array<f32>;
@group(0) @binding(5) var<storage, read_write> fi: array<i32>;
@group(0) @binding(6) var<storage, read_write> fu: array<u32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = (x[i] << n[i]) + (x[i] >> n[i]) * 3u;
        w[i] = x[i] + x[i] * 2u;
        fi[i] = i32(f[i]);
        fu[i] = u32(f[i]);
    }
}
