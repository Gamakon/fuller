// grid 16 16 1 1
// The store to h as a left-nested add chain (2026-10-12): h[c] = h[c] + a + b, read as h[c] += a; h[c] += b (bit-exact).
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
        f1 = f1 + phi[t * cfg.d + gr];
    }
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + f1 * 0.0;
    }
}
