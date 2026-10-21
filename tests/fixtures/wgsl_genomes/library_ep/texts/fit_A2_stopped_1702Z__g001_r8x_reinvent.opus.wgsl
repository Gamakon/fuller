// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0);
    var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u;
    var i9: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; var i15: u32 = 0u; var i16: u32 = 0u; var i17: u32 = 0u;
    var b0: bool = false; var b1: bool = false; var b2: bool = false; var b3: bool = false; var b4: bool = false;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    pw[((k) & 15u)] = vec4<f32>(0.0f);
    }
    b0 = (lidx < 64u);
    i2 = ((lidx) & (63u));
    i3 = (((i2) & (15u)) * 4u);
    i4 = ((i2) >> ((4u) & 31u));
    if (b0) {
    i5 = ((wid.x * 64u) + i3);
    b1 = (i5 < cfg.d);
    b2 = ((i5 + 1u) < cfg.d);
    b3 = ((i5 + 2u) < cfg.d);
    b4 = ((i5 + 3u) < cfg.d);
    } else {
    i6 = ((wid.y * 64u) + i3);
    i5 = (cfg.r_off + i6);
    b1 = ((i6 < cfg.r_rows) && (i5 < cfg.d));
    b2 = (((i6 + 1u) < cfg.r_rows) && ((i5 + 1u) < cfg.d));
    b3 = (((i6 + 2u) < cfg.r_rows) && ((i5 + 2u) < cfg.d));
    b4 = (((i6 + 3u) < cfg.r_rows) && ((i5 + 3u) < cfg.d));
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i9 = (t + (k * 32u));
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i12 = (i4 + (k2 * 4u));
    i13 = (i9 + i12);
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    if ((i13 < cfg.m)) {
    i14 = ((i13 * cfg.d) + i5);
    if (b1) {
    f0 = phi[i14];
    }
    if (b2) {
    f1 = phi[(i14 + 1u)];
    }
    if (b3) {
    f2 = phi[(i14 + 2u)];
    }
    if (b4) {
    f3 = phi[(i14 + 3u)];
    }
    }
    i15 = ((((i12 * 16u) + ((i2) & (15u)))) & (511u));
    if (b0) {
    wg_va[((i15) & 511u)] = vec4<f32>(f0, f1, f2, f3);
    } else {
    wg_vb[((i15) & 511u)] = vec4<f32>(f0, f1, f2, f3);
    }
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
    i16 = (k2 * 32u);
    i17 = (i16 + (lid.x * 2u));
    v8 = wg_va[((i17) & 511u)];
    v9 = wg_va[(((i17 + 1u)) & 511u)];
    v12 = wg_va[(((i17 + 16u)) & 511u)];
    v13 = wg_va[(((i17 + 17u)) & 511u)];
    v10 = wg_vb[(((i16 + lid.y)) & 511u)];
    v11 = wg_vb[((((i16 + 16u) + lid.y)) & 511u)];
    v0 = (v0 + ((vec4<f32>(v10[0u]) * v8) + (vec4<f32>(v11[0u]) * v12)));
    v1 = (v1 + ((vec4<f32>(v10[0u]) * v9) + (vec4<f32>(v11[0u]) * v13)));
    v2 = (v2 + ((vec4<f32>(v10[1u]) * v8) + (vec4<f32>(v11[1u]) * v12)));
    v3 = (v3 + ((vec4<f32>(v10[1u]) * v9) + (vec4<f32>(v11[1u]) * v13)));
    v4 = (v4 + ((vec4<f32>(v10[2u]) * v8) + (vec4<f32>(v11[2u]) * v12)));
    v5 = (v5 + ((vec4<f32>(v10[2u]) * v9) + (vec4<f32>(v11[2u]) * v13)));
    v6 = (v6 + ((vec4<f32>(v10[3u]) * v8) + (vec4<f32>(v11[3u]) * v12)));
    v7 = (v7 + ((vec4<f32>(v10[3u]) * v9) + (vec4<f32>(v11[3u]) * v13)));
    }
    pw[0u] = (pw[0u] + v0);
    pw[1u] = (pw[1u] + v1);
    pw[2u] = (pw[2u] + v2);
    pw[3u] = (pw[3u] + v3);
    pw[4u] = (pw[4u] + v4);
    pw[5u] = (pw[5u] + v5);
    pw[6u] = (pw[6u] + v6);
    pw[7u] = (pw[7u] + v7);
    workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    pv[((k) & 15u)] = (pv[((k) & 15u)] + pw[((k) & 15u)]);
    pw[((k) & 15u)] = vec4<f32>(0.0f);
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    if ((((gid.x * 8u) + k2) < cfg.d)) {
    h[((((gid.y * 4u) + k)) * cfg.d + ((gid.x * 8u) + k2))] = h[((((gid.y * 4u) + k)) * cfg.d + ((gid.x * 8u) + k2))] + pv[(((k * 2u) + (k2 >> 2u)) & 15u)][(k2) & 3u];
    }
    }
    }
    }
}
