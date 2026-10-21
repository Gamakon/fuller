// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0);
    var i1: u32 = 0u; var i2: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    i1 = 0u;
    for (k = 0u; k < 2u; k = k + 1u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    if (((t + (((lidx + (256u * k))) >> ((4u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) < cfg.d)) {
    v0[(0u) & 3u] = phi[(((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)))];
    }
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 1u) < cfg.d)) {
    v0[(1u) & 3u] = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 1u)];
    }
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 2u) < cfg.d)) {
    v0[(2u) & 3u] = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 2u)];
    }
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 3u) < cfg.d)) {
    v0[(3u) & 3u] = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 3u)];
    }
    if (((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) < cfg.d))) {
    v1[(0u) & 3u] = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)))];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 1u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 1u) < cfg.d))) {
    v1[(1u) & 3u] = phi[(((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 1u)];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 2u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 2u) < cfg.d))) {
    v1[(2u) & 3u] = phi[(((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 2u)];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 3u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 3u) < cfg.d))) {
    v1[(3u) & 3u] = phi[(((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 3u)];
    }
    }
    wg_va[(((lidx + (256u * k))) & 511u)] = v0;
    wg_vb[(((lidx + (256u * k))) & 511u)] = v1;
    }
    workgroupBarrier();
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    v4 = vec4<f32>(0.0f);
    v5 = vec4<f32>(0.0f);
    v6 = vec4<f32>(0.0f);
    v7 = vec4<f32>(0.0f);
    v8 = vec4<f32>(0.0f);
    v9 = vec4<f32>(0.0f);
    i1 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    v2 = (v2 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].x) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v3 = (v3 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].y) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v4 = (v4 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].z) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v5 = (v5 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].w) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v6 = (v6 + (vec4<f32>(wg_vb[((((((2u * k) + 1u) * 16u) + lid.y)) & 511u)].x) * wg_va[((((((2u * k) + 1u) * 16u) + lid.x)) & 511u)]));
    v7 = (v7 + (vec4<f32>(wg_vb[((((((2u * k) + 1u) * 16u) + lid.y)) & 511u)].y) * wg_va[((((((2u * k) + 1u) * 16u) + lid.x)) & 511u)]));
    v8 = (v8 + (vec4<f32>(wg_vb[((((((2u * k) + 1u) * 16u) + lid.y)) & 511u)].z) * wg_va[((((((2u * k) + 1u) * 16u) + lid.x)) & 511u)]));
    v9 = (v9 + (vec4<f32>(wg_vb[((((((2u * k) + 1u) * 16u) + lid.y)) & 511u)].w) * wg_va[((((((2u * k) + 1u) * 16u) + lid.x)) & 511u)]));
    }
    v10 = (v10 + (v2 + v6));
    v11 = (v11 + (v3 + v7));
    v12 = (v12 + (v4 + v8));
    v13 = (v13 + (v5 + v9));
    if ((((((t) >> ((5u) & 31u))) & (3u)) == 3u)) {
    v14 = (v14 + v10);
    v15 = (v15 + v11);
    v16 = (v16 + v12);
    v17 = (v17 + v13);
    v10 = vec4<f32>(0.0f);
    v11 = vec4<f32>(0.0f);
    v12 = vec4<f32>(0.0f);
    v13 = vec4<f32>(0.0f);
    }
    workgroupBarrier();
    }
    pv[0u] = (v14 + v10);
    pv[1u] = (v15 + v11);
    pv[2u] = (v16 + v12);
    pv[3u] = (v17 + v13);
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d))) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pv[((k) & 15u)][(k2) & 3u]; }
    }
    }
    }
    }
}
