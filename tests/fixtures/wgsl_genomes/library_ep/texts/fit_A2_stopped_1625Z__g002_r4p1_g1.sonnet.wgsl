// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0; 
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0); 
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); 
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0); 
    var i0: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i2 = (t + (k * 32u));
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i4 = (lidx + (128u * k2));
    i5 = (i2 + ((i4) >> ((4u) & 31u)));
    i6 = (((i4) & (15u)) * 4u);
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    if ((i5 < cfg.m)) {
    i7 = (((i5 * cfg.d) + (wid.x * 64u)) + i6);
    if ((((wid.x * 64u) + i6) < cfg.d)) {
    f0 = phi[i7];
    }
    if (((((wid.x * 64u) + i6) + 1u) < cfg.d)) {
    f1 = phi[(i7 + 1u)];
    }
    if (((((wid.x * 64u) + i6) + 2u) < cfg.d)) {
    f2 = phi[(i7 + 2u)];
    }
    if (((((wid.x * 64u) + i6) + 3u) < cfg.d)) {
    f3 = phi[(i7 + 3u)];
    }
    i7 = ((((i5 * cfg.d) + cfg.r_off) + (wid.y * 64u)) + i6);
    if (((((wid.y * 64u) + i6) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + i6) < cfg.d))) {
    f4 = phi[i7];
    }
    if ((((((wid.y * 64u) + i6) + 1u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i6) + 1u) < cfg.d))) {
    f5 = phi[(i7 + 1u)];
    }
    if ((((((wid.y * 64u) + i6) + 2u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i6) + 2u) < cfg.d))) {
    f6 = phi[(i7 + 2u)];
    }
    if ((((((wid.y * 64u) + i6) + 3u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i6) + 3u) < cfg.d))) {
    f7 = phi[(i7 + 3u)];
    }
    }
    wg_va[((i4) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[((i4) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f);
    v5 = vec4<f32>(0.0f);
    v6 = vec4<f32>(0.0f);
    v7 = vec4<f32>(0.0f);
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    i8 = ((k2 * 32u) + (lid.x * 2u));
    v8 = wg_va[((i8) & 511u)];
    v9 = wg_va[(((i8 + 1u)) & 511u)];
    v10 = wg_va[(((i8 + 16u)) & 511u)];
    v11 = wg_va[(((i8 + 17u)) & 511u)];
    v12 = wg_vb[((((k2 * 32u) + lid.y)) & 511u)];
    v13 = wg_vb[(((((k2 * 32u) + 16u) + lid.y)) & 511u)];
    v0 = (v0 + ((vec4<f32>(v12[(0u) & 3u]) * v8) + (vec4<f32>(v13[(0u) & 3u]) * v10)));
    v1 = (v1 + ((vec4<f32>(v12[(0u) & 3u]) * v9) + (vec4<f32>(v13[(0u) & 3u]) * v11)));
    v2 = (v2 + ((vec4<f32>(v12[(1u) & 3u]) * v8) + (vec4<f32>(v13[(1u) & 3u]) * v10)));
    v3 = (v3 + ((vec4<f32>(v12[(1u) & 3u]) * v9) + (vec4<f32>(v13[(1u) & 3u]) * v11)));
    v4 = (v4 + ((vec4<f32>(v12[(2u) & 3u]) * v8) + (vec4<f32>(v13[(2u) & 3u]) * v10)));
    v5 = (v5 + ((vec4<f32>(v12[(2u) & 3u]) * v9) + (vec4<f32>(v13[(2u) & 3u]) * v11)));
    v6 = (v6 + ((vec4<f32>(v12[(3u) & 3u]) * v8) + (vec4<f32>(v13[(3u) & 3u]) * v10)));
    v7 = (v7 + ((vec4<f32>(v12[(3u) & 3u]) * v9) + (vec4<f32>(v13[(3u) & 3u]) * v11)));
    }
    v14 = (v14 + v0);
    v15 = (v15 + v1);
    v16 = (v16 + v2);
    v17 = (v17 + v3);
    v18 = (v18 + v4);
    v19 = (v19 + v5);
    v20 = (v20 + v6);
    v21 = (v21 + v7);
    workgroupBarrier();
    }
    }
    pv[0u] = (pv[0u] + v14);
    pv[1u] = (pv[1u] + v15);
    pv[2u] = (pv[2u] + v16);
    pv[3u] = (pv[3u] + v17);
    pv[4u] = (pv[4u] + v18);
    pv[5u] = (pv[5u] + v19);
    pv[6u] = (pv[6u] + v20);
    pv[7u] = (pv[7u] + v21);
    v14 = vec4<f32>(0.0f);
    v15 = vec4<f32>(0.0f);
    v16 = vec4<f32>(0.0f);
    v17 = vec4<f32>(0.0f);
    v18 = vec4<f32>(0.0f);
    v19 = vec4<f32>(0.0f);
    v20 = vec4<f32>(0.0f);
    v21 = vec4<f32>(0.0f);
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + (lid.y * 4u)) + k) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (lid.y * 4u)) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + (lid.x * 8u)) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 8u) + min(k2, 7u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 8u) + min(k2, 7u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 8u) + min(k2, 7u))))] + pv[((((k * 2u) + ((k2) >> ((2u) & 31u)))) & 15u)][(k2) & 3u]; }
    }
    }
    }
    }
}
