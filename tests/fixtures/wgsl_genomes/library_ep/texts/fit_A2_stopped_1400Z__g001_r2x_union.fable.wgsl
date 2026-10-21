// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var va: vec4<f32> = vec4<f32>(0.0); var vb: vec4<f32> = vec4<f32>(0.0); var vc: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 64u) {
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    i0 = t + (k3 * 16u);
    for (k = 0u; k < 2u; k = k + 1u) {
    i1 = lidx + (128u * k);
    i2 = i0 + (i1 >> 4u);
    i3 = (i1 & 15u) * 4u;
    f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
    f4 = 0.0f; f5 = 0.0f; f6 = 0.0f; f7 = 0.0f;
    if (i2 < cfg.m) {
    i4 = (i2 * cfg.d) + (wid.x * 64u) + i3;
    if (((wid.x * 64u) + i3) < cfg.d) { f0 = phi[i4]; }
    if (((wid.x * 64u) + i3 + 1u) < cfg.d) { f1 = phi[i4 + 1u]; }
    if (((wid.x * 64u) + i3 + 2u) < cfg.d) { f2 = phi[i4 + 2u]; }
    if (((wid.x * 64u) + i3 + 3u) < cfg.d) { f3 = phi[i4 + 3u]; }
    i4 = (i2 * cfg.d) + cfg.r_off + (wid.y * 64u) + i3;
    if ((((wid.y * 64u) + i3) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i3) < cfg.d)) { f4 = phi[i4]; }
    if ((((wid.y * 64u) + i3 + 1u) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i3 + 1u) < cfg.d)) { f5 = phi[i4 + 1u]; }
    if ((((wid.y * 64u) + i3 + 2u) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i3 + 2u) < cfg.d)) { f6 = phi[i4 + 2u]; }
    if ((((wid.y * 64u) + i3 + 3u) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i3 + 3u) < cfg.d)) { f7 = phi[i4 + 3u]; }
    }
    wg_va[(i1 & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[(i1 & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
    for (k = 0u; k < 16u; k = k + 1u) {
    va = wg_va[(((k * 16u) + (lid.x * 2u)) & 511u)];
    vb = wg_va[(((k * 16u) + (lid.x * 2u) + 1u) & 511u)];
    vc = wg_vb[(((k * 16u) + lid.y) & 511u)];
    v0 = v0 + (vec4<f32>(vc.x) * va);
    v1 = v1 + (vec4<f32>(vc.x) * vb);
    v2 = v2 + (vec4<f32>(vc.y) * va);
    v3 = v3 + (vec4<f32>(vc.y) * vb);
    v4 = v4 + (vec4<f32>(vc.z) * va);
    v5 = v5 + (vec4<f32>(vc.z) * vb);
    v6 = v6 + (vec4<f32>(vc.w) * va);
    v7 = v7 + (vec4<f32>(vc.w) * vb);
    }
    v8 = v8 + v0; v9 = v9 + v1; v10 = v10 + v2; v11 = v11 + v3;
    v12 = v12 + v4; v13 = v13 + v5; v14 = v14 + v6; v15 = v15 + v7;
    workgroupBarrier();
    }
    pv[((lit_u32(0u)) & 15u)] = pv[((lit_u32(0u)) & 15u)] + v8;
    pv[((lit_u32(1u)) & 15u)] = pv[((lit_u32(1u)) & 15u)] + v9;
    pv[((lit_u32(2u)) & 15u)] = pv[((lit_u32(2u)) & 15u)] + v10;
    pv[((lit_u32(3u)) & 15u)] = pv[((lit_u32(3u)) & 15u)] + v11;
    pv[((lit_u32(4u)) & 15u)] = pv[((lit_u32(4u)) & 15u)] + v12;
    pv[((lit_u32(5u)) & 15u)] = pv[((lit_u32(5u)) & 15u)] + v13;
    pv[((lit_u32(6u)) & 15u)] = pv[((lit_u32(6u)) & 15u)] + v14;
    pv[((lit_u32(7u)) & 15u)] = pv[((lit_u32(7u)) & 15u)] + v15;
    v8 = vec4<f32>(0.0f); v9 = vec4<f32>(0.0f); v10 = vec4<f32>(0.0f); v11 = vec4<f32>(0.0f);
    v12 = vec4<f32>(0.0f); v13 = vec4<f32>(0.0f); v14 = vec4<f32>(0.0f); v15 = vec4<f32>(0.0f);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((wid.y * 64u) + (lid.y * 4u)) + k) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (lid.y * 4u)) + k)) < cfg.d)) {
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    if ((((wid.x * 64u) + (lid.x * 8u)) + k2) < cfg.d) {
    if ((((wid.y * 64u + lid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((wid.x * 64u + lid.x * 8u) + min(k2, 7u))) < cfg.d && (cfg.r_off + (((wid.y * 64u + lid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((wid.y * 64u + lid.y * 4u) + min(k, 3u))) * cfg.d + (((wid.x * 64u + lid.x * 8u) + min(k2, 7u))))] = h[((((wid.y * 64u + lid.y * 4u) + min(k, 3u))) * cfg.d + (((wid.x * 64u + lid.x * 8u) + min(k2, 7u))))] + pv[(((k * 2u) + (k2 >> 2u)) & 15u)][(k2) & 3u]; }
    }
    }
    }
    }
}
