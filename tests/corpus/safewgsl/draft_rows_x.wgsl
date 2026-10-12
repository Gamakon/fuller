// grid 64 4 2 2 rows-x
// The layout knob (2026-10-12): the output's rows on thread x (the fast index), its columns on y; a 2 x 2 block per thread.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(64, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc: array<f32, 4>;
    let r0 = gid.x * 2u;
    let c0 = gid.y * 2u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                acc[i * 2u + j] = acc[i * 2u + j] + phi[t * cfg.d + min(cfg.r_off + r0 + i, cfg.d - 1u)] * phi[t * cfg.d + min(c0 + j, cfg.d - 1u)];
            }
        }
    }
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            if (r0 + i < cfg.r_rows && c0 + j < cfg.d && cfg.r_off + r0 + i < cfg.d) {
                h[(r0 + i) * cfg.d + c0 + j] = h[(r0 + i) * cfg.d + c0 + j] + acc[i * 2u + j];
            }
        }
    }
}
