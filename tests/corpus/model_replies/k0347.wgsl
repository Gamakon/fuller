// grid 16 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0);
    var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0); var v22: vec4<f32> = vec4<f32>(0.0); var v23: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    i0 = wid.x * 128u;
    i1 = wid.y * 64u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pv[k & 15u] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 64u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i2 = t + k * 16u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i3 = lidx + 128u * k2;
    i4 = i2 + (i3 >> 5u);
    i5 = i0 + (i3 & 31u) * 4u;
    f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
    if (i4 < cfg.m) {
    i6 = i4 * cfg.d + i5;
    if (i5 < cfg.d) { f0 = phi[i6]; }
    if ((i5 + 1u) < cfg.d) { f1 = phi[i6 + 1u]; }
    if ((i5 + 2u) < cfg.d) { f2 = phi[i6 + 2u]; }
    if ((i5 + 3u) < cfg.d) { f3 = phi[i6 + 3u]; }
    }
    wg_va[i3 & 511u] = vec4<f32>(f0, f1, f2, f3);
    }
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    i3 = lidx + 128u * k2;
    i4 = i2 + (i3 >> 4u);
    i5 = i1 + (i3 & 15u) * 4u;
    f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
    if (i4 < cfg.m) {
    i6 = i4 * cfg.d + cfg.r_off + i5;
    if (i5 < cfg.r_rows && (cfg.r_off + i5) < cfg.d) { f0 = phi[i6]; }
    if ((i5 + 1u) < cfg.r_rows && (cfg.r_off + i5 + 1u) < cfg.d) { f1 = phi[i6 + 1u]; }
    if ((i5 + 2u) < cfg.r_rows && (cfg.r_off + i5 + 2u) < cfg.d) { f2 = phi[i6 + 2u]; }
    if ((i5 + 3u) < cfg.r_rows && (cfg.r_off + i5 + 3u) < cfg.d) { f3 = phi[i6 + 3u]; }
    }
    wg_vb[i3 & 511u] = vec4<f32>(f0, f1, f2, f3);
    }
    workgroupBarrier();
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i7 = k2 * 64u;
    i8 = k2 * 32u;
    v16 = wg_va[(i7 + lid.x) & 511u];
    v17 = wg_va[(i7 + 16u + lid.x) & 511u];
    v18 = wg_va[(i7 + 32u + lid.x) & 511u];
    v19 = wg_va[(i7 + 48u + lid.x) & 511u];
    v20 = wg_vb[(i8 + lid.y) & 511u];
    v21 = wg_vb[(i8 + 8u + lid.y) & 511u];
    v22 = wg_vb[(i8 + 16u + lid.y) & 511u];
    v23 = wg_vb[(i8 + 24u + lid.y) & 511u];
    v0 = v0 + ((vec4<f32>(v20.x) * v16) + (vec4<f32>(v22.x) * v18));
    v1 = v1 + ((vec4<f32>(v20.x) * v17) + (vec4<f32>(v22.x) * v19));
    v2 = v2 + ((vec4<f32>(v20.y) * v16) + (vec4<f32>(v22.y) * v18));
    v3 = v3 + ((vec4<f32>(v20.y) * v17) + (vec4<f32>(v22.y) * v19));
    v4 = v4 + ((vec4<f32>(v20.z) * v16) + (vec4<f32>(v22.z) * v18));
    v5 = v5 + ((vec4<f32>(v20.z) * v17) + (vec4<f32>(v22.z) * v19));
    v6 = v6 + ((vec4<f32>(v20.w) * v16) + (vec4<f32>(v22.w) * v18));
    v7 = v7 + ((vec4<f32>(v20.w) * v17) + (vec4<f32>(v22.w) * v19));
    v8 = v8 + ((vec4<f32>(v21.x) * v16) + (vec4<f32>(v23.x) * v18));
    v9 = v9 + ((vec4<f32>(v21.x) * v17) + (vec4<f32>(v23.x) * v19));
    v10 = v10 + ((vec4<f32>(v21.y) * v16) + (vec4<f32>(v23.y) * v18));
    v11 = v11 + ((vec4<f32>(v21.y) * v17) + (vec4<f32>(v23.y) * v19));
    v12 = v12 + ((vec4<f32>(v21.z) * v16) + (vec4<f32>(v23.z) * v18));
    v13 = v13 + ((vec4<f32>(v21.z) * v17) + (vec4<f32>(v23.z) * v19));
    v14 = v14 + ((vec4<f32>(v21.w) * v16) + (vec4<f32>(v23.w) * v18));
    v15 = v15 + ((vec4<f32>(v21.w) * v17) + (vec4<f32>(v23.w) * v19));
    }
    workgroupBarrier();
    }
    pv[0u] = pv[0u] + v0; pv[1u] = pv[1u] + v1; pv[2u] = pv[2u] + v2; pv[3u] = pv[3u] + v3;
    pv[4u] = pv[4u] + v4; pv[5u] = pv[5u] + v5; pv[6u] = pv[6u] + v6; pv[7u] = pv[7u] + v7;
    pv[8u] = pv[8u] + v8; pv[9u] = pv[9u] + v9; pv[10u] = pv[10u] + v10; pv[11u] = pv[11u] + v11;
    pv[12u] = pv[12u] + v12; pv[13u] = pv[13u] + v13; pv[14u] = pv[14u] + v14; pv[15u] = pv[15u] + v15;
    v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
    v8 = vec4<f32>(0.0f); v9 = vec4<f32>(0.0f); v10 = vec4<f32>(0.0f); v11 = vec4<f32>(0.0f);
    v12 = vec4<f32>(0.0f); v13 = vec4<f32>(0.0f); v14 = vec4<f32>(0.0f); v15 = vec4<f32>(0.0f);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    i9 = i1 + lid.y * 4u + (k & 3u) + (k >> 2u) * 32u;
    if (i9 < cfg.r_rows && (cfg.r_off + i9) < cfg.d) {
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    i10 = i0 + lid.x * 4u + k2 * 64u + k3;
    if (i10 < cfg.d) {
    h[i9 * cfg.d + i10] = h[i9 * cfg.d + i10] + pv[(k * 2u + k2) & 15u][k3 & 3u];
    }
    }
    }
    }
    }
}
