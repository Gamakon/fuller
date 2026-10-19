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
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0); 
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; 
    var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; var i15: u32 = 0u; 
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u; var k4: u32 = 0u;
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
    i6 = (t + (k * 32u));
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i10 = (lidx + (128u * k2));
    i12 = (i6 + ((i10) >> ((4u) & 31u)));
    i11 = (((i10) & (15u)) * 4u);
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    if ((i12 < cfg.m)) {
    i13 = (((i12 * cfg.d) + (wid.x * 64u)) + i11);
    if ((((wid.x * 64u) + i11) < cfg.d)) {
    f0 = phi[i13];
    }
    if (((((wid.x * 64u) + i11) + 1u) < cfg.d)) {
    f1 = phi[(i13 + 1u)];
    }
    if (((((wid.x * 64u) + i11) + 2u) < cfg.d)) {
    f2 = phi[(i13 + 2u)];
    }
    if (((((wid.x * 64u) + i11) + 3u) < cfg.d)) {
    f3 = phi[(i13 + 3u)];
    }
    i13 = ((((i12 * cfg.d) + cfg.r_off) + (wid.y * 64u)) + i11);
    if (((((wid.y * 64u) + i11) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + i11) < cfg.d))) {
    f4 = phi[i13];
    }
    if ((((((wid.y * 64u) + i11) + 1u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i11) + 1u) < cfg.d))) {
    f5 = phi[(i13 + 1u)];
    }
    if ((((((wid.y * 64u) + i11) + 2u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i11) + 2u) < cfg.d))) {
    f6 = phi[(i13 + 2u)];
    }
    if ((((((wid.y * 64u) + i11) + 3u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i11) + 3u) < cfg.d))) {
    f7 = phi[(i13 + 3u)];
    }
    }
    wg_va[((i10) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[((i10) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
    }
    workgroupBarrier();
    v13 = vec4<f32>(0.0f);
    v14 = vec4<f32>(0.0f);
    v15 = vec4<f32>(0.0f);
    v16 = vec4<f32>(0.0f);
    v17 = vec4<f32>(0.0f);
    v8 = vec4<f32>(0.0f);
    v18 = vec4<f32>(0.0f);
    v19 = vec4<f32>(0.0f);
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    i14 = (k2 * 32u);
    i15 = (i14 + (lid.y * 2u));
    v10 = wg_va[(((i14 + lid.x)) & 511u)];
    v12 = wg_va[((((i14 + 16u) + lid.x)) & 511u)];
    v20 = wg_vb[((i15) & 511u)];
    v9 = wg_vb[(((i15 + 1u)) & 511u)];
    v21 = wg_vb[(((i15 + 16u)) & 511u)];
    v11 = wg_vb[(((i15 + 17u)) & 511u)];
    v13 = (v13 + ((vec4<f32>(v20[(0u) & 3u]) * v10) + (vec4<f32>(v21[(0u) & 3u]) * v12)));
    v14 = (v14 + ((vec4<f32>(v20[(1u) & 3u]) * v10) + (vec4<f32>(v21[(1u) & 3u]) * v12)));
    v15 = (v15 + ((vec4<f32>(v20[(2u) & 3u]) * v10) + (vec4<f32>(v21[(2u) & 3u]) * v12)));
    v16 = (v16 + ((vec4<f32>(v20[(3u) & 3u]) * v10) + (vec4<f32>(v21[(3u) & 3u]) * v12)));
    v17 = (v17 + ((vec4<f32>(v9[(0u) & 3u]) * v10) + (vec4<f32>(v11[(0u) & 3u]) * v12)));
    v8 = (v8 + ((vec4<f32>(v9[(1u) & 3u]) * v10) + (vec4<f32>(v11[(1u) & 3u]) * v12)));
    v18 = (v18 + ((vec4<f32>(v9[(2u) & 3u]) * v10) + (vec4<f32>(v11[(2u) & 3u]) * v12)));
    v19 = (v19 + ((vec4<f32>(v9[(3u) & 3u]) * v10) + (vec4<f32>(v11[(3u) & 3u]) * v12)));
    }
    v0 = (v0 + v13);
    v1 = (v1 + v14);
    v2 = (v2 + v15);
    v3 = (v3 + v16);
    v4 = (v4 + v17);
    v5 = (v5 + v8);
    v6 = (v6 + v18);
    v7 = (v7 + v19);
    workgroupBarrier();
    }
    i10 = (((((t) >> ((7u) & 31u))) & (1u)) * 8u);
    pv[((i10) & 15u)] = (pv[((i10) & 15u)] + v0);
    pv[(((i10 + 1u)) & 15u)] = (pv[(((i10 + 1u)) & 15u)] + v1);
    pv[(((i10 + 2u)) & 15u)] = (pv[(((i10 + 2u)) & 15u)] + v2);
    pv[(((i10 + 3u)) & 15u)] = (pv[(((i10 + 3u)) & 15u)] + v3);
    pv[(((i10 + 4u)) & 15u)] = (pv[(((i10 + 4u)) & 15u)] + v4);
    pv[(((i10 + 5u)) & 15u)] = (pv[(((i10 + 5u)) & 15u)] + v5);
    pv[(((i10 + 6u)) & 15u)] = (pv[(((i10 + 6u)) & 15u)] + v6);
    pv[(((i10 + 7u)) & 15u)] = (pv[(((i10 + 7u)) & 15u)] + v7);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    if (((((gid.y * 8u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 8u) + k)) < cfg.d))) {
    v13 = (pv[((k) & 15u)] + pv[(((k + 8u)) & 15u)]);
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + v13[(k2) & 3u]; }
    }
    }
    }
    }
}
