// grid 16 16 1 1
// Integer atomics in workgroup memory (2026-10-12): every row form once; one cell per thread.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> hits: array<atomic<u32>, 256>;
var<workgroup> tally: array<atomic<i32>, 64>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var n: u32 = 0u;
    let gr = cfg.r_off + gid.y;
    atomicStore(&hits[lidx], 0u);
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
        n = atomicAdd(&hits[lidx], 1u);
        atomicOr(&hits[lidx], 0u);
        atomicAdd(&tally[lidx & 63u], -1i);
    }
    n = atomicLoad(&hits[lidx]) + n - n;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0;
    }
}
