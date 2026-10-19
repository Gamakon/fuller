// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var i2: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
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
    v8 = vec4<f32>(0.0f);
    v9 = vec4<f32>(0.0f);
    v10 = vec4<f32>(0.0f);
    v11 = vec4<f32>(0.0f);
    v12 = vec4<f32>(0.0f);
    v13 = vec4<f32>(0.0f);
    v14 = vec4<f32>(0.0f);
    v15 = vec4<f32>(0.0f);
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    i8 = (k2 * 32u);
    i9 = (i8 + (lid.y * 2u));
    v16 = wg_va[(((i8 + lid.x)) & 511u)];
    v17 = wg_va[((((i8 + 16u) + lid.x)) & 511u)];
    v18 = wg_vb[((i9) & 511u)];
    v19 = wg_vb[(((i9 + 1u)) & 511u)];
    v20 = wg_vb[(((i9 + 16u)) & 511u)];
    v21 = wg_vb[(((i9 + 17u)) & 511u)];
    v8 = (v8 + ((vec4<f32>(v18[(0u) & 3u]) * v16) + (vec4<f32>(v20[(0u) & 3u]) * v17)));
    v9 = (v9 + ((vec4<f32>(v18[(1u) & 3u]) * v16) + (vec4<f32>(v20[(1u) & 3u]) * v17)));
    v10 = (v10 + ((vec4<f32>(v18[(2u) & 3u]) * v16) + (vec4<f32>(v20[(2u) & 3u]) * v17)));
    v11 = (v11 + ((vec4<f32>(v18[(3u) & 3u]) * v16) + (vec4<f32>(v20[(3u) & 3u]) * v17)));
    v12 = (v12 + ((vec4<f32>(v19[(0u) & 3u]) * v16) + (vec4<f32>(v21[(0u) & 3u]) * v17)));
    v13 = (v13 + ((vec4<f32>(v19[(1u) & 3u]) * v16) + (vec4<f32>(v21[(1u) & 3u]) * v17)));
    v14 = (v14 + ((vec4<f32>(v19[(2u) & 3u]) * v16) + (vec4<f32>(v21[(2u) & 3u]) * v17)));
    v15 = (v15 + ((vec4<f32>(v19[(3u) & 3u]) * v16) + (vec4<f32>(v21[(3u) & 3u]) * v17)));
    }
    v0 = (v0 + v8);
    v1 = (v1 + v9);
    v2 = (v2 + v10);
    v3 = (v3 + v11);
    v4 = (v4 + v12);
    v5 = (v5 + v13);
    v6 = (v6 + v14);
    v7 = (v7 + v15);
    workgroupBarrier();
    }
    i4 = (((((t) >> ((7u) & 31u))) & (1u)) * 8u);
    pv[((i4) & 15u)] = (pv[((i4) & 15u)] + v0);
    pv[(((i4 + 1u)) & 15u)] = (pv[(((i4 + 1u)) & 15u)] + v1);
    pv[(((i4 + 2u)) & 15u)] = (pv[(((i4 + 2u)) & 15u)] + v2);
    pv[(((i4 + 3u)) & 15u)] = (pv[(((i4 + 3u)) & 15u)] + v3);
    pv[(((i4 + 4u)) & 15u)] = (pv[(((i4 + 4u)) & 15u)] + v4);
    pv[(((i4 + 5u)) & 15u)] = (pv[(((i4 + 5u)) & 15u)] + v5);
    pv[(((i4 + 6u)) & 15u)] = (pv[(((i4 + 6u)) & 15u)] + v6);
    pv[(((i4 + 7u)) & 15u)] = (pv[(((i4 + 7u)) & 15u)] + v7);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    if (((((gid.y * 8u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 8u) + k)) < cfg.d))) {
    v8 = (pv[((k) & 15u)] + pv[(((k + 8u)) & 15u)]);
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + v8[(k2) & 3u]; }
    }
    }
    }
    }
}
