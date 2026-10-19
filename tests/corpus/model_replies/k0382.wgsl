// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u;
    var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    i0 = (wid.x * 128u);
    i1 = (wid.y * 64u);
    i2 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    i3 = 0u;
    for (t = 0u; t < cfg.m; t = t + 128u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f);
    v5 = vec4<f32>(0.0f);
    v6 = vec4<f32>(0.0f);
    v7 = vec4<f32>(0.0f);
    i2 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i4 = (t + (k * 16u));
    i5 = 0u;
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    i6 = (lidx + (256u * k2));
    i7 = (i4 + ((i6) >> ((5u) & 31u)));
    i8 = (i0 + (((i6) & (31u)) * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i7 < cfg.m)) {
    i9 = ((i7 * cfg.d) + i8);
    if ((i8 < cfg.d)) {
    f0 = phi[i9];
    }
    if (((i8 + 1u) < cfg.d)) {
    f1 = phi[(i9 + 1u)];
    }
    if (((i8 + 2u) < cfg.d)) {
    f2 = phi[(i9 + 2u)];
    }
    if (((i8 + 3u) < cfg.d)) {
    f3 = phi[(i9 + 3u)];
    }
    }
    wg_va[((i6) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    }
    i6 = lidx;
    i7 = (i4 + ((i6) >> ((4u) & 31u)));
    i8 = (i1 + (((i6) & (15u)) * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i7 < cfg.m)) {
    i9 = (((i7 * cfg.d) + cfg.r_off) + i8);
    if (((i8 < cfg.r_rows) && ((cfg.r_off + i8) < cfg.d))) {
    f0 = phi[i9];
    }
    if ((((i8 + 1u) < cfg.r_rows) && (((cfg.r_off + i8) + 1u) < cfg.d))) {
    f1 = phi[(i9 + 1u)];
    }
    if ((((i8 + 2u) < cfg.r_rows) && (((cfg.r_off + i8) + 2u) < cfg.d))) {
    f2 = phi[(i9 + 2u)];
    }
    if ((((i8 + 3u) < cfg.r_rows) && (((cfg.r_off + i8) + 3u) < cfg.d))) {
    f3 = phi[(i9 + 3u)];
    }
    }
    wg_vb[((i6) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    workgroupBarrier();
    v8 = vec4<f32>(0.0f);
    v9 = vec4<f32>(0.0f);
    v10 = vec4<f32>(0.0f);
    v11 = vec4<f32>(0.0f);
    v12 = vec4<f32>(0.0f);
    v13 = vec4<f32>(0.0f);
    v14 = vec4<f32>(0.0f);
    v15 = vec4<f32>(0.0f);
    i5 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i10 = (k2 * 64u);
    i11 = ((k2 * 32u) + (lid.y * 2u));
    v16 = wg_va[(((i10 + lid.x)) & 511u)];
    v17 = wg_va[((((i10 + 32u) + lid.x)) & 511u)];
    v18 = wg_vb[((i11) & 511u)];
    v19 = wg_vb[(((i11 + 1u)) & 511u)];
    v20 = wg_vb[(((i11 + 16u)) & 511u)];
    v21 = wg_vb[(((i11 + 17u)) & 511u)];
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
    i12 = (((((t) >> ((7u) & 31u))) & (1u)) * 8u);
    pv[((i12) & 15u)] = (pv[((i12) & 15u)] + v0);
    pv[(((i12 + 1u)) & 15u)] = (pv[(((i12 + 1u)) & 15u)] + v1);
    pv[(((i12 + 2u)) & 15u)] = (pv[(((i12 + 2u)) & 15u)] + v2);
    pv[(((i12 + 3u)) & 15u)] = (pv[(((i12 + 3u)) & 15u)] + v3);
    pv[(((i12 + 4u)) & 15u)] = (pv[(((i12 + 4u)) & 15u)] + v4);
    pv[(((i12 + 5u)) & 15u)] = (pv[(((i12 + 5u)) & 15u)] + v5);
    pv[(((i12 + 6u)) & 15u)] = (pv[(((i12 + 6u)) & 15u)] + v6);
    pv[(((i12 + 7u)) & 15u)] = (pv[(((i12 + 7u)) & 15u)] + v7);
    }
    i2 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i13 = ((i1 + (lid.y * 8u)) + k);
    if (((i13 < cfg.r_rows) && ((cfg.r_off + i13) < cfg.d))) {
    v8 = (pv[((k) & 15u)] + pv[(((k + 8u)) & 15u)]);
    i5 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i14 = ((i0 + (lid.x * 4u)) + k2);
    if ((i14 < cfg.d)) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + v8[(k2) & 3u]; }
    }
    }
    }
    }
}
