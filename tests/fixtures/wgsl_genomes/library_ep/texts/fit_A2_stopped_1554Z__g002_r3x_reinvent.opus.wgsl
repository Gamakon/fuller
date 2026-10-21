// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0);
    var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[k] = vec4<f32>(0.0f);
    pw[k] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    i6 = (lidx + (256u * k));
    i7 = (t + (i6 >> 5u));
    i8 = ((wid.x * 128u) + ((i6 & 31u) * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i7 < cfg.m)) {
    i10 = (i7 * cfg.d);
    if ((i8 < cfg.d)) {
    f0 = phi[(i10 + i8)];
    }
    if (((i8 + 1u) < cfg.d)) {
    f1 = phi[((i10 + i8) + 1u)];
    }
    if (((i8 + 2u) < cfg.d)) {
    f2 = phi[((i10 + i8) + 2u)];
    }
    if (((i8 + 3u) < cfg.d)) {
    f3 = phi[((i10 + i8) + 3u)];
    }
    }
    wg_va[(i6 & 511u)] = vec4<f32>(f0, f1, f2, f3);
    }
    i7 = (t + (lidx >> 4u));
    i9 = ((wid.y * 64u) + ((lidx & 15u) * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i7 < cfg.m)) {
    i10 = (i7 * cfg.d);
    if (((i9 < cfg.r_rows) && ((cfg.r_off + i9) < cfg.d))) {
    f0 = phi[((i10 + cfg.r_off) + i9)];
    }
    if ((((i9 + 1u) < cfg.r_rows) && (((cfg.r_off + i9) + 1u) < cfg.d))) {
    f1 = phi[(((i10 + cfg.r_off) + i9) + 1u)];
    }
    if ((((i9 + 2u) < cfg.r_rows) && (((cfg.r_off + i9) + 2u) < cfg.d))) {
    f2 = phi[(((i10 + cfg.r_off) + i9) + 2u)];
    }
    if ((((i9 + 3u) < cfg.r_rows) && (((cfg.r_off + i9) + 3u) < cfg.d))) {
    f3 = phi[(((i10 + cfg.r_off) + i9) + 3u)];
    }
    }
    wg_vb[(lidx & 511u)] = vec4<f32>(f0, f1, f2, f3);
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f);
    v5 = vec4<f32>(0.0f);
    v6 = vec4<f32>(0.0f);
    v7 = vec4<f32>(0.0f);
    for (k = 0u; k < 16u; k = k + 1u) {
    v8 = wg_va[(((k * 32u) + lid.x) & 511u)];
    v9 = wg_va[((((k * 32u) + 16u) + lid.x) & 511u)];
    v10 = wg_vb[(((k * 16u) + lid.y) & 511u)];
    v0 = (v0 + (vec4<f32>(v10.x) * v8));
    v1 = (v1 + (vec4<f32>(v10.x) * v9));
    v2 = (v2 + (vec4<f32>(v10.y) * v8));
    v3 = (v3 + (vec4<f32>(v10.y) * v9));
    v4 = (v4 + (vec4<f32>(v10.z) * v8));
    v5 = (v5 + (vec4<f32>(v10.z) * v9));
    v6 = (v6 + (vec4<f32>(v10.w) * v8));
    v7 = (v7 + (vec4<f32>(v10.w) * v9));
    }
    pv[0u] = (pv[0u] + v0);
    pv[1u] = (pv[1u] + v1);
    pv[2u] = (pv[2u] + v2);
    pv[3u] = (pv[3u] + v3);
    pv[4u] = (pv[4u] + v4);
    pv[5u] = (pv[5u] + v5);
    pv[6u] = (pv[6u] + v6);
    pv[7u] = (pv[7u] + v7);
    if ((((t >> 4u) & 31u) == 31u)) {
    pw[0u] = (pw[0u] + pv[0u]);
    pw[1u] = (pw[1u] + pv[1u]);
    pw[2u] = (pw[2u] + pv[2u]);
    pw[3u] = (pw[3u] + pv[3u]);
    pw[4u] = (pw[4u] + pv[4u]);
    pw[5u] = (pw[5u] + pv[5u]);
    pw[6u] = (pw[6u] + pv[6u]);
    pw[7u] = (pw[7u] + pv[7u]);
    pv[0u] = vec4<f32>(0.0f);
    pv[1u] = vec4<f32>(0.0f);
    pv[2u] = vec4<f32>(0.0f);
    pv[3u] = vec4<f32>(0.0f);
    pv[4u] = vec4<f32>(0.0f);
    pv[5u] = vec4<f32>(0.0f);
    pv[6u] = vec4<f32>(0.0f);
    pv[7u] = vec4<f32>(0.0f);
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    pw[k] = (pw[k] + pv[k]);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    i11 = (((wid.y * 64u) + (lid.y * 4u)) + k);
    if (((i11 < cfg.r_rows) && ((cfg.r_off + i11) < cfg.d))) {
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    i12 = ((((wid.x * 128u) + (k2 * 64u)) + (lid.x * 4u)) + k3);
    if ((i12 < cfg.d)) {
    h[((i11 * cfg.d) + i12)] = h[((i11 * cfg.d) + i12)] + pw[(((k * 2u) + k2) & 15u)][(k3) & 3u];
    }
    }
    }
    }
    }
}
