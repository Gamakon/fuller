// Integer division and remainder at WGSL's edges, eight lanes in one
// invocation: x / 0 == x, x % 0 == 0, i32::MIN / -1 == i32::MIN,
// i32::MIN % -1 == 0, truncation toward zero for negatives.
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> q: array<i32>;
@group(0) @binding(3) var<storage, read_write> r: array<i32>;
@group(0) @binding(4) var<storage, read> ua: array<u32>;
@group(0) @binding(5) var<storage, read> ub: array<u32>;
@group(0) @binding(6) var<storage, read_write> uq: array<u32>;
@group(0) @binding(7) var<storage, read_write> ur: array<u32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 8u; i = i + 1u) {
        q[i] = a[i] / b[i];
        r[i] = a[i] % b[i];
        uq[i] = ua[i] / ub[i];
        ur[i] = ua[i] % ub[i];
    }
}
