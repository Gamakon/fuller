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
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0; 
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0); 
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); 
    var i1: u32 = 0u; var i2: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    if (((t + (((lidx + (256u * k))) >> ((4u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) < cfg.d)) {
    f0 = phi[(((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)))];
    }
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 1u) < cfg.d)) {
    f1 = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 1u)];
    }
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 2u) < cfg.d)) {
    f2 = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 2u)];
    }
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 3u) < cfg.d)) {
    f3 = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + ((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 3u)];
    }
    if (((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) < cfg.d))) {
    f4 = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)))];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 1u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 1u) < cfg.d))) {
    f5 = phi[(((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 1u)];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 2u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 2u) < cfg.d))) {
    f6 = phi[(((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 2u)];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + 3u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 3u) < cfg.d))) {
    f7 = phi[(((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u))) + 3u)];
    }
    }
    wg_va[(((lidx + (256u * k))) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[(((lidx + (256u * k))) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
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
    for (k = 0u; k < 16u; k = k + 1u) {
    v0 = (v0 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].x) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v1 = (v1 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].y) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v2 = (v2 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].z) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v3 = (v3 + (vec4<f32>(wg_vb[(((((2u * k) * 16u) + lid.y)) & 511u)].w) * wg_va[(((((2u * k) * 16u) + lid.x)) & 511u)]));
    v4 = (v4 + (vec4<f32>(wg_vb[((((((2u * k) * 16u) + 16u) + lid.y)) & 511u)].x) * wg_va[((((((2u * k) * 16u) + 16u) + lid.x)) & 511u)]));
    v5 = (v5 + (vec4<f32>(wg_vb[((((((2u * k) * 16u) + 16u) + lid.y)) & 511u)].y) * wg_va[((((((2u * k) * 16u) + 16u) + lid.x)) & 511u)]));
    v6 = (v6 + (vec4<f32>(wg_vb[((((((2u * k) * 16u) + 16u) + lid.y)) & 511u)].z) * wg_va[((((((2u * k) * 16u) + 16u) + lid.x)) & 511u)]));
    v7 = (v7 + (vec4<f32>(wg_vb[((((((2u * k) * 16u) + 16u) + lid.y)) & 511u)].w) * wg_va[((((((2u * k) * 16u) + 16u) + lid.x)) & 511u)]));
    }
    if ((((((t) >> ((5u) & 31u))) & (1u)) == 0u)) {
    v8 = (v8 + (v0 + v4));
    v9 = (v9 + (v1 + v5));
    v10 = (v10 + (v2 + v6));
    v11 = (v11 + (v3 + v7));
    } else {
    v12 = (v12 + (v0 + v4));
    v13 = (v13 + (v1 + v5));
    v14 = (v14 + (v2 + v6));
    v15 = (v15 + (v3 + v7));
    }
    workgroupBarrier();
    }
    pv[0u] = (v8 + v12);
    pv[1u] = (v9 + v13);
    pv[2u] = (v10 + v14);
    pv[3u] = (v11 + v15);
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pv[((k) & 15u)][(k2) & 3u]; }
    }
    }
    }
    }
}
