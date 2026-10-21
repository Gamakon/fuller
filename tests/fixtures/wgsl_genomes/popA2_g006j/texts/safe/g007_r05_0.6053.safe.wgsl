// grid 8 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { var t = v; return t[i & 3u]; }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pv: array<vec4<f32>, 16>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    var pk: array<u32, 16>;
    var pm: array<bool, 16>;
    var pn: array<bool, 16>;
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(vec3<f32>(vec2<f32>(0.0f, 0.0f), 0.0f), 0.0f);
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    pi[((k) & 15u)] = ((wid.y * 64u + lid.y) + min(k, 7u) * 8u);
    pm[((k) & 15u)] = ((((wid.y * 64u + lid.y) + min(k, 7u) * 8u) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 64u + lid.y) + min(k, 7u) * 8u)) < cfg.d));
    pj[((k) & 15u)] = (cfg.r_off + ((wid.y * 64u + lid.y) + min(k, 7u) * 8u));
    pn[((k) & 15u)] = (((wid.x * 64u + lid.x) + min(k, 7u) * 8u) < cfg.d);
    pk[((k) & 15u)] = ((wid.x * 64u + lid.x) + min(k, 7u) * 8u);
    }
    i1 = 0u;
    for (t = 0u; t < cfg.m; t = t + 1u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    if (pm[((k) & 15u)]) {
    f0 = phi[((t * cfg.d) + pj[((k) & 15u)])];
    }
    pa[((k) & 63u)] = f0;
    f1 = 0.0f;
    if (pn[((k) & 15u)]) {
    f1 = phi[((t * cfg.d) + pk[((k) & 15u)])];
    }
    pb[((k) & 63u)] = f1;
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[(((k * 2u)) & 15u)] = (pv[(((k * 2u)) & 15u)] + (vec4<f32>(pa[((k) & 63u)]) * vec4<f32>(vec3<f32>(vec2<f32>(pb[0u], pb[1u]), pb[2u]), pb[3u])));
    pv[((((k * 2u) + 1u)) & 15u)] = (pv[((((k * 2u) + 1u)) & 15u)] + (vec4<f32>(pa[((k) & 63u)]) * vec4<f32>(vec3<f32>(vec2<f32>(pb[4u], pb[5u]), pb[6u]), pb[7u])));
    }
    }
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    if (pm[((k) & 15u)]) {
    if (pn[((k2) & 15u)]) {
    if (((((wid.y * 64u + lid.y) + min(k, 7u) * 8u)) < cfg.r_rows) & ((((wid.x * 64u + lid.x) + min(k2, 7u) * 8u)) < cfg.d) & ((cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 7u) * 8u))) < cfg.d)) { h[((((wid.y * 64u + lid.y) + min(k, 7u) * 8u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 7u) * 8u)))] = h[((((wid.y * 64u + lid.y) + min(k, 7u) * 8u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 7u) * 8u)))] + at_vec4f(pv[((((k * 2u) + ((k2) >> ((2u) & 31u)))) & 15u)], ((k2) & (3u))); }
    }
    }
    }
    }
}
