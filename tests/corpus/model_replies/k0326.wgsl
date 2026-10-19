// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0; 
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0); 
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); 
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0); var v22: vec4<f32> = vec4<f32>(0.0); var v23: vec4<f32> = vec4<f32>(0.0); 
    var v24: vec4<f32> = vec4<f32>(0.0); var v25: vec4<f32> = vec4<f32>(0.0); var v26: vec4<f32> = vec4<f32>(0.0); var v27: vec4<f32> = vec4<f32>(0.0); var v28: vec4<f32> = vec4<f32>(0.0); var v29: vec4<f32> = vec4<f32>(0.0); var v30: vec4<f32> = vec4<f32>(0.0); var v31: vec4<f32> = vec4<f32>(0.0); 
    var i14: u32 = 0u; var i22: u32 = 0u; var i23: u32 = 0u; var i24: u32 = 0u; var i25: u32 = 0u; var i26: u32 = 0u; var i27: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i14 = (t + (k * 32u));
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i22 = (lidx + (128u * k2));
    i23 = (i14 + ((i22) >> ((4u) & 31u)));
    i24 = (((i22) & (15u)) * 4u);
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    if ((i23 < cfg.m)) {
    i25 = (((i23 * cfg.d) + (wid.x * 64u)) + i24);
    if ((((wid.x * 64u) + i24) < cfg.d)) {
    f0 = phi[i25];
    }
    if (((((wid.x * 64u) + i24) + 1u) < cfg.d)) {
    f1 = phi[(i25 + 1u)];
    }
    if (((((wid.x * 64u) + i24) + 2u) < cfg.d)) {
    f2 = phi[(i25 + 2u)];
    }
    if (((((wid.x * 64u) + i24) + 3u) < cfg.d)) {
    f3 = phi[(i25 + 3u)];
    }
    i25 = ((((i23 * cfg.d) + cfg.r_off) + (wid.y * 64u)) + i24);
    if (((((wid.y * 64u) + i24) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + i24) < cfg.d))) {
    f4 = phi[i25];
    }
    if ((((((wid.y * 64u) + i24) + 1u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i24) + 1u) < cfg.d))) {
    f5 = phi[(i25 + 1u)];
    }
    if ((((((wid.y * 64u) + i24) + 2u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i24) + 2u) < cfg.d))) {
    f6 = phi[(i25 + 2u)];
    }
    if ((((((wid.y * 64u) + i24) + 3u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i24) + 3u) < cfg.d))) {
    f7 = phi[(i25 + 3u)];
    }
    }
    wg_va[((i22) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[((i22) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
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
    i26 = (k2 * 32u);
    i27 = (i26 + (lid.y * 2u));
    v8 = wg_va[(((i26 + lid.x)) & 511u)];
    v9 = wg_va[((((i26 + 16u) + lid.x)) & 511u)];
    v10 = wg_vb[((i27) & 511u)];
    v11 = wg_vb[(((i27 + 1u)) & 511u)];
    v12 = wg_vb[(((i27 + 16u)) & 511u)];
    v13 = wg_vb[(((i27 + 17u)) & 511u)];
    v0 = (v0 + ((vec4<f32>(v10[(0u) & 3u]) * v8) + (vec4<f32>(v12[(0u) & 3u]) * v9)));
    v1 = (v1 + ((vec4<f32>(v10[(1u) & 3u]) * v8) + (vec4<f32>(v12[(1u) & 3u]) * v9)));
    v2 = (v2 + ((vec4<f32>(v10[(2u) & 3u]) * v8) + (vec4<f32>(v12[(2u) & 3u]) * v9)));
    v3 = (v3 + ((vec4<f32>(v10[(3u) & 3u]) * v8) + (vec4<f32>(v12[(3u) & 3u]) * v9)));
    v4 = (v4 + ((vec4<f32>(v11[(0u) & 3u]) * v8) + (vec4<f32>(v13[(0u) & 3u]) * v9)));
    v5 = (v5 + ((vec4<f32>(v11[(1u) & 3u]) * v8) + (vec4<f32>(v13[(1u) & 3u]) * v9)));
    v6 = (v6 + ((vec4<f32>(v11[(2u) & 3u]) * v8) + (vec4<f32>(v13[(2u) & 3u]) * v9)));
    v7 = (v7 + ((vec4<f32>(v11[(3u) & 3u]) * v8) + (vec4<f32>(v13[(3u) & 3u]) * v9)));
    }
    if (((k) & 1u) == 0u) {
    v14 = (v14 + v0);
    v15 = (v15 + v1);
    v16 = (v16 + v2);
    v17 = (v17 + v3);
    v18 = (v18 + v4);
    v19 = (v19 + v5);
    v20 = (v20 + v6);
    v21 = (v21 + v7);
    } else {
    v22 = (v22 + v0);
    v23 = (v23 + v1);
    v24 = (v24 + v2);
    v25 = (v25 + v3);
    v26 = (v26 + v4);
    v27 = (v27 + v5);
    v28 = (v28 + v6);
    v29 = (v29 + v7);
    }
    workgroupBarrier();
    }
    pv[0u] = (pv[0u] + (v14 + v22));
    pv[1u] = (pv[1u] + (v15 + v23));
    pv[2u] = (pv[2u] + (v16 + v24));
    pv[3u] = (pv[3u] + (v17 + v25));
    pv[4u] = (pv[4u] + (v18 + v26));
    pv[5u] = (pv[5u] + (v19 + v27));
    pv[6u] = (pv[6u] + (v20 + v28));
    pv[7u] = (pv[7u] + (v21 + v29));
    v14 = vec4<f32>(0.0f);
    v15 = vec4<f32>(0.0f);
    v16 = vec4<f32>(0.0f);
    v17 = vec4<f32>(0.0f);
    v18 = vec4<f32>(0.0f);
    v19 = vec4<f32>(0.0f);
    v20 = vec4<f32>(0.0f);
    v21 = vec4<f32>(0.0f);
    v22 = vec4<f32>(0.0f);
    v23 = vec4<f32>(0.0f);
    v24 = vec4<f32>(0.0f);
    v25 = vec4<f32>(0.0f);
    v26 = vec4<f32>(0.0f);
    v27 = vec4<f32>(0.0f);
    v28 = vec4<f32>(0.0f);
    v29 = vec4<f32>(0.0f);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    if (((((gid.y * 8u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 8u) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pv[((k) & 15u)][(k2) & 3u]; }
    }
    }
    }
    }
}
