// grid 16 16 1 1
// Workgroup arrays as knobs (2026-10-12): a vec2 array padded to 1,056 elements
// and an f32 array of 256, each read back at its own length; one cell per thread.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> stage: array<vec2<f32>, 1056>;
var<workgroup> scale: array<f32, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        stage[lidx] = vec2<f32>(phi[t * cfg.d + gr], phi[t * cfg.d + gid.x]);
        scale[lidx & 255u] = 1.0;
        let p = stage[lidx];
        f0 = f0 + p.x * p.y * scale[lidx & 255u];
    }
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0;
    }
}
