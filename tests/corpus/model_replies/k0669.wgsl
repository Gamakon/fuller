// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var v10: vec4<f32> = vec4<f32>(0.0);
    var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0);
    var v13: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
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
    for (k = 0u; k < 32u; k = k + 1u) {
    v2 = (v2 + (vec4<f32>(wg_vb[((((k * 16u) + lid.y)) & 511u)].x) * wg_va[((((k * 16u) + lid.x)) & 511u)]));
    v3 = (v3 + (vec4<f32>(wg_vb[((((k * 16u) + lid.y)) & 511u)].y) * wg_va[((((k * 16u) + lid.x)) & 511u)]));
    v4 = (v4 + (vec4<f32>(wg_vb[((((k * 16u) + lid.y)) & 511u)].z) * wg_va[((((k * 16u) + lid.x)) & 511u)]));
    v5 = (v5 + (vec4<f32>(wg_vb[((((k * 16u) + lid.y)) & 511u)].w) * wg_va[((((k * 16u) + lid.x)) & 511u)]));
    }
    if ((((((t) >> ((5u) & 31u))) & (1u)) == 0u)) {
    v6 = (v6 + v2);
    v7 = (v7 + v3);
    v8 = (v8 + v4);
    v9 = (v9 + v5);
    } else {
    v10 = (v10 + v2);
    v11 = (v11 + v3);
    v12 = (v12 + v4);
    v13 = (v13 + v5);
    }
    workgroupBarrier();
    }
    pv[0u] = (v6 + v10);
    pv[1u] = (v7 + v11);
    pv[2u] = (v8 + v12);
    pv[3u] = (v9 + v13);
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    h[(((gid.y * 4u) + k) * cfg.d + ((gid.x * 4u) + k2))] = h[(((gid.y * 4u) + k) * cfg.d + ((gid.x * 4u) + k2))] + pv[((k) & 15u)][(k2) & 3u];
    }
    }
    }
    }
}
