// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0);
    var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var i6: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; var i15: u32 = 0u; var i16: u32 = 0u; var i17: u32 = 0u; var i18: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    pw[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f);
    v5 = vec4<f32>(0.0f);
    v6 = vec4<f32>(0.0f);
    v7 = vec4<f32>(0.0f);
    for (k = 0u; k < 8u; k = k + 1u) {
    i6 = (t + (k * 16u));
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    i13 = (lidx + (256u * k2));
    i14 = (i6 + (i13 >> 5u));
    i15 = (i13 & 31u);
    i17 = ((wid.x * 128u) + (i15 * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i14 < cfg.m)) {
    i16 = ((i14 * cfg.d) + i17);
    if ((i17 < cfg.d)) {
    f0 = phi[i16];
    }
    if (((i17 + 1u) < cfg.d)) {
    f1 = phi[(i16 + 1u)];
    }
    if (((i17 + 2u) < cfg.d)) {
    f2 = phi[(i16 + 2u)];
    }
    if (((i17 + 3u) < cfg.d)) {
    f3 = phi[(i16 + 3u)];
    }
    }
    wg_va[(((((i13 >> 5u) * 32u) + (i15 >> 1u)) + ((i15 & 1u) * 16u)) & 511u)] = vec4<f32>(f0, f1, f2, f3);
    }
    i14 = (i6 + (lidx >> 4u));
    i15 = (lidx & 15u);
    i17 = ((wid.y * 64u) + (i15 * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i14 < cfg.m)) {
    i16 = (((i14 * cfg.d) + cfg.r_off) + i17);
    if (((i17 < cfg.r_rows) && ((cfg.r_off + i17) < cfg.d))) {
    f0 = phi[i16];
    }
    if ((((i17 + 1u) < cfg.r_rows) && (((cfg.r_off + i17) + 1u) < cfg.d))) {
    f1 = phi[(i16 + 1u)];
    }
    if ((((i17 + 2u) < cfg.r_rows) && (((cfg.r_off + i17) + 2u) < cfg.d))) {
    f2 = phi[(i16 + 2u)];
    }
    if ((((i17 + 3u) < cfg.r_rows) && (((cfg.r_off + i17) + 3u) < cfg.d))) {
    f3 = phi[(i16 + 3u)];
    }
    }
    wg_vb[((lidx) & 511u)] = vec4<f32>(f0, f1, f2, f3);
    workgroupBarrier();
    v8 = vec4<f32>(0.0f);
    v9 = vec4<f32>(0.0f);
    pw[0u] = vec4<f32>(0.0f);
    pw[1u] = vec4<f32>(0.0f);
    pw[2u] = vec4<f32>(0.0f);
    pw[3u] = vec4<f32>(0.0f);
    pw[4u] = vec4<f32>(0.0f);
    pw[5u] = vec4<f32>(0.0f);
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i17 = (k2 * 64u);
    i18 = (k2 * 32u);
    v16 = wg_va[((i17 + lid.x) & 511u)];
    v17 = wg_va[(((i17 + 16u) + lid.x) & 511u)];
    v18 = wg_va[(((i17 + 32u) + lid.x) & 511u)];
    v19 = wg_va[(((i17 + 48u) + lid.x) & 511u)];
    v20 = wg_vb[((i18 + lid.y) & 511u)];
    v21 = wg_vb[(((i18 + 16u) + lid.y) & 511u)];
    v8 = (v8 + ((vec4<f32>(v20.x) * v16) + (vec4<f32>(v21.x) * v18)));
    v9 = (v9 + ((vec4<f32>(v20.y) * v16) + (vec4<f32>(v21.y) * v18)));
    pw[0u] = (pw[0u] + ((vec4<f32>(v20.z) * v16) + (vec4<f32>(v21.z) * v18)));
    pw[1u] = (pw[1u] + ((vec4<f32>(v20.w) * v16) + (vec4<f32>(v21.w) * v18)));
    pw[2u] = (pw[2u] + ((vec4<f32>(v20.x) * v17) + (vec4<f32>(v21.x) * v19)));
    pw[3u] = (pw[3u] + ((vec4<f32>(v20.y) * v17) + (vec4<f32>(v21.y) * v19)));
    pw[4u] = (pw[4u] + ((vec4<f32>(v20.z) * v17) + (vec4<f32>(v21.z) * v19)));
    pw[5u] = (pw[5u] + ((vec4<f32>(v20.w) * v17) + (vec4<f32>(v21.w) * v19)));
    }
    v0 = (v0 + v8);
    v1 = (v1 + v9);
    v2 = (v2 + pw[0u]);
    v3 = (v3 + pw[1u]);
    v4 = (v4 + pw[2u]);
    v5 = (v5 + pw[3u]);
    v6 = (v6 + pw[4u]);
    v7 = (v7 + pw[5u]);
    workgroupBarrier();
    }
    i16 = (((t >> 7u) & 1u) * 8u);
    pv[((i16) & 15u)] = (pv[((i16) & 15u)] + v0);
    pv[((i16 + 1u) & 15u)] = (pv[((i16 + 1u) & 15u)] + v1);
    pv[((i16 + 2u) & 15u)] = (pv[((i16 + 2u) & 15u)] + v2);
    pv[((i16 + 3u) & 15u)] = (pv[((i16 + 3u) & 15u)] + v3);
    pv[((i16 + 4u) & 15u)] = (pv[((i16 + 4u) & 15u)] + v4);
    pv[((i16 + 5u) & 15u)] = (pv[((i16 + 5u) & 15u)] + v5);
    pv[((i16 + 6u) & 15u)] = (pv[((i16 + 6u) & 15u)] + v6);
    pv[((i16 + 7u) & 15u)] = (pv[((i16 + 7u) & 15u)] + v7);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    i13 = ((gid.y * 4u) + k);
    if (((i13 < cfg.r_rows) && ((cfg.r_off + i13) < cfg.d))) {
    v8 = (pv[((k) & 15u)] + pv[((k + 8u) & 15u)]);
    v9 = (pv[((k + 4u) & 15u)] + pv[((k + 12u) & 15u)]);
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i14 = ((gid.x * 8u) + k2);
    if ((i14 < cfg.d)) {
    h[((i13 * cfg.d) + i14)] = h[((i13 * cfg.d) + i14)] + v8[(k2) & 3u];
    }
    i15 = ((gid.x * 8u) + (k2 + 4u));
    if ((i15 < cfg.d)) {
    h[((i13 * cfg.d) + i15)] = h[((i13 * cfg.d) + i15)] + v9[(k2) & 3u];
    }
    }
    }
    }
}
