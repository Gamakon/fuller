// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    if (((t + (((lidx + (256u * k))) >> ((4u) & 31u))) < cfg.m)) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + k2) < cfg.d)) {
    v0[(k2) & 3u] = phi[(((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + (((wid.x * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + k2))];
    }
    if ((((((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + k2) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + k2)) < cfg.d))) {
    v1[(k2) & 3u] = phi[((((t + (((lidx + (256u * k))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + (((wid.y * 64u) + ((((lidx + (256u * k))) & (15u)) * 4u)) + k2))];
    }
    }
    }
    wg_va[(((lidx + (256u * k))) & 511u)] = v0;
    wg_vb[(((lidx + (256u * k))) & 511u)] = v1;
    }
    workgroupBarrier();
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    f8 = 0.0f;
    f9 = 0.0f;
    f10 = 0.0f;
    f11 = 0.0f;
    f12 = 0.0f;
    f13 = 0.0f;
    f14 = 0.0f;
    f15 = 0.0f;
    for (k = 0u; k < 32u; k = k + 1u) {
    f0 = (f0 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].x * wg_va[((((k * 16u) + lid.x)) & 511u)].x));
    f1 = (f1 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].x * wg_va[((((k * 16u) + lid.x)) & 511u)].y));
    f2 = (f2 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].x * wg_va[((((k * 16u) + lid.x)) & 511u)].z));
    f3 = (f3 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].x * wg_va[((((k * 16u) + lid.x)) & 511u)].w));
    f4 = (f4 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].y * wg_va[((((k * 16u) + lid.x)) & 511u)].x));
    f5 = (f5 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].y * wg_va[((((k * 16u) + lid.x)) & 511u)].y));
    f6 = (f6 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].y * wg_va[((((k * 16u) + lid.x)) & 511u)].z));
    f7 = (f7 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].y * wg_va[((((k * 16u) + lid.x)) & 511u)].w));
    f8 = (f8 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].z * wg_va[((((k * 16u) + lid.x)) & 511u)].x));
    f9 = (f9 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].z * wg_va[((((k * 16u) + lid.x)) & 511u)].y));
    f10 = (f10 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].z * wg_va[((((k * 16u) + lid.x)) & 511u)].z));
    f11 = (f11 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].z * wg_va[((((k * 16u) + lid.x)) & 511u)].w));
    f12 = (f12 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].w * wg_va[((((k * 16u) + lid.x)) & 511u)].x));
    f13 = (f13 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].w * wg_va[((((k * 16u) + lid.x)) & 511u)].y));
    f14 = (f14 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].w * wg_va[((((k * 16u) + lid.x)) & 511u)].z));
    f15 = (f15 + (wg_vb[((((k * 16u) + lid.y)) & 511u)].w * wg_va[((((k * 16u) + lid.x)) & 511u)].w));
    }
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 0u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 0u)) & 63u)] + f0);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 1u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 1u)) & 63u)] + f1);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 2u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 2u)) & 63u)] + f2);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 3u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 3u)) & 63u)] + f3);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 4u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 4u)) & 63u)] + f4);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 5u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 5u)) & 63u)] + f5);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 6u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 6u)) & 63u)] + f6);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 7u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 7u)) & 63u)] + f7);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 8u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 8u)) & 63u)] + f8);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 9u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 9u)) & 63u)] + f9);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 10u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 10u)) & 63u)] + f10);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 11u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 11u)) & 63u)] + f11);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 12u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 12u)) & 63u)] + f12);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 13u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 13u)) & 63u)] + f13);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 14u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 14u)) & 63u)] + f14);
    pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 15u)) & 63u)] = (pa[((((((((t) >> ((5u) & 31u))) & (1u)) * 16u) + 15u)) & 63u)] + f15);
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + (lid.y * 4u)) + k) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (lid.y * 4u)) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + (lid.x * 4u)) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + (pa[((((k * 4u) + k2)) & 63u)] + pa[(((((k * 4u) + k2) + 16u)) & 63u)]); }
    }
    }
    }
    }
}
