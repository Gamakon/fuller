// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0);
    var i2: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 128u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i2 = t + (k * 32u);
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i4 = lidx + (128u * k2);
    i5 = i2 + (i4 >> 4u);
    i6 = (i4 & 15u) * 4u;
    f0 = 0.0;
    f1 = 0.0;
    f2 = 0.0;
    f3 = 0.0;
    f4 = 0.0;
    f5 = 0.0;
    f6 = 0.0;
    f7 = 0.0;
    if (i5 < cfg.m) {
    i7 = (i5 * cfg.d) + (wid.x * 64u) + i6;
    if (((wid.x * 64u) + i6) < cfg.d) {
    f0 = phi[i7];
    }
    if (((wid.x * 64u) + i6 + 1u) < cfg.d) {
    f1 = phi[i7 + 1u];
    }
    if (((wid.x * 64u) + i6 + 2u) < cfg.d) {
    f2 = phi[i7 + 2u];
    }
    if (((wid.x * 64u) + i6 + 3u) < cfg.d) {
    f3 = phi[i7 + 3u];
    }
    i7 = (i5 * cfg.d) + cfg.r_off + (wid.y * 64u) + i6;
    if ((((wid.y * 64u) + i6) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i6) < cfg.d)) {
    f4 = phi[i7];
    }
    if ((((wid.y * 64u) + i6 + 1u) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i6 + 1u) < cfg.d)) {
    f5 = phi[i7 + 1u];
    }
    if ((((wid.y * 64u) + i6 + 2u) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i6 + 2u) < cfg.d)) {
    f6 = phi[i7 + 2u];
    }
    if ((((wid.y * 64u) + i6 + 3u) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + i6 + 3u) < cfg.d)) {
    f7 = phi[i7 + 3u];
    }
    }
    wg_va[i4 & 511u] = vec4<f32>(f0, f1, f2, f3);
    wg_vb[i4 & 511u] = vec4<f32>(f4, f5, f6, f7);
    }
    workgroupBarrier();
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    i8 = k2 * 32u;
    i9 = i8 + (lid.y * 2u);
    v8 = wg_va[(i8 + lid.x) & 511u];
    v9 = wg_va[(i8 + 16u + lid.x) & 511u];
    v10 = wg_vb[i9 & 511u];
    v11 = wg_vb[(i9 + 1u) & 511u];
    v12 = wg_vb[(i9 + 16u) & 511u];
    v13 = wg_vb[(i9 + 17u) & 511u];
    v0 = v0 + ((v10.x * v8) + (v12.x * v9));
    v1 = v1 + ((v10.y * v8) + (v12.y * v9));
    v2 = v2 + ((v10.z * v8) + (v12.z * v9));
    v3 = v3 + ((v10.w * v8) + (v12.w * v9));
    v4 = v4 + ((v11.x * v8) + (v13.x * v9));
    v5 = v5 + ((v11.y * v8) + (v13.y * v9));
    v6 = v6 + ((v11.z * v8) + (v13.z * v9));
    v7 = v7 + ((v11.w * v8) + (v13.w * v9));
    }
    workgroupBarrier();
    }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((((gid.y * 8u) + k) < cfg.r_rows) && ((cfg.r_off + (gid.y * 8u) + k) < cfg.d)) {
    v14 = v0;
    if (k == 1u) { v14 = v1; }
    if (k == 2u) { v14 = v2; }
    if (k == 3u) { v14 = v3; }
    if (k == 4u) { v14 = v4; }
    if (k == 5u) { v14 = v5; }
    if (k == 6u) { v14 = v6; }
    if (k == 7u) { v14 = v7; }
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((gid.x * 4u) + k2) < cfg.d) {
    h[(((gid.y * 8u) + k) * cfg.d) + (gid.x * 4u) + k2] = h[(((gid.y * 8u) + k) * cfg.d) + (gid.x * 4u) + k2] + v14[k2];
    }
    }
    }
    }
}
