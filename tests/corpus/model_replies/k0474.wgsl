// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0; 
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0; 
    var f16r: f32 = 0.0; var f17: f32 = 0.0;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    f16r = 0.0f;
    f17 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (256u * k))) >> ((((lit_u32(6u)) & (lit_u32(31u)))) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u))) < cfg.d)) {
    f16r = phi[((((t + (((((lid.y * 16u) + lid.x) + (256u * k))) >> ((((lit_u32(6u)) & (lit_u32(31u)))) & 31u))) * cfg.d) + (wid.x * 64u)) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u)))];
    }
    if (((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u))) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u))) < cfg.d))) {
    f17 = phi[(((((t + (((((lid.y * 16u) + lid.x) + (256u * k))) >> ((((lit_u32(6u)) & (lit_u32(31u)))) & 31u))) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + (((((lid.y * 16u) + lid.x) + (256u * k))) & (63u)))];
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (256u * k))) & 2047u)] = f16r;
    wg_b[(((((lid.y * 16u) + lid.x) + (256u * k))) & 2047u)] = f17;
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
    for (k = 0u; k < 16u; k = k + 1u) {
    f0 = (f0 + (wg_b[((((k * 64u) + lid.y)) & 2047u)] * wg_a[((((k * 64u) + lid.x)) & 2047u)]));
    f1 = (f1 + (wg_b[((((k * 64u) + lid.y)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    f2 = (f2 + (wg_b[((((k * 64u) + lid.y)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    f3 = (f3 + (wg_b[((((k * 64u) + lid.y)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    f4 = (f4 + (wg_b[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_a[((((k * 64u) + lid.x)) & 2047u)]));
    f5 = (f5 + (wg_b[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    f6 = (f6 + (wg_b[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    f7 = (f7 + (wg_b[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    f8 = (f8 + (wg_b[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_a[((((k * 64u) + lid.x)) & 2047u)]));
    f9 = (f9 + (wg_b[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    f10 = (f10 + (wg_b[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    f11 = (f11 + (wg_b[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    f12 = (f12 + (wg_b[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_a[((((k * 64u) + lid.x)) & 2047u)]));
    f13 = (f13 + (wg_b[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    f14 = (f14 + (wg_b[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    f15 = (f15 + (wg_b[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    }
    pa[((lit_u32(0u)) & 63u)] = (pa[((lit_u32(0u)) & 63u)] + f0);
    pa[((lit_u32(1u)) & 63u)] = (pa[((lit_u32(1u)) & 63u)] + f1);
    pa[((lit_u32(2u)) & 63u)] = (pa[((lit_u32(2u)) & 63u)] + f2);
    pa[((lit_u32(3u)) & 63u)] = (pa[((lit_u32(3u)) & 63u)] + f3);
    pa[((lit_u32(4u)) & 63u)] = (pa[((lit_u32(4u)) & 63u)] + f4);
    pa[((lit_u32(5u)) & 63u)] = (pa[((lit_u32(5u)) & 63u)] + f5);
    pa[((lit_u32(6u)) & 63u)] = (pa[((lit_u32(6u)) & 63u)] + f6);
    pa[((lit_u32(7u)) & 63u)] = (pa[((lit_u32(7u)) & 63u)] + f7);
    pa[((lit_u32(8u)) & 63u)] = (pa[((lit_u32(8u)) & 63u)] + f8);
    pa[((lit_u32(9u)) & 63u)] = (pa[((lit_u32(9u)) & 63u)] + f9);
    pa[((lit_u32(10u)) & 63u)] = (pa[((lit_u32(10u)) & 63u)] + f10);
    pa[((lit_u32(11u)) & 63u)] = (pa[((lit_u32(11u)) & 63u)] + f11);
    pa[((lit_u32(12u)) & 63u)] = (pa[((lit_u32(12u)) & 63u)] + f12);
    pa[((lit_u32(13u)) & 63u)] = (pa[((lit_u32(13u)) & 63u)] + f13);
    pa[((lit_u32(14u)) & 63u)] = (pa[((lit_u32(14u)) & 63u)] + f14);
    pa[((lit_u32(15u)) & 63u)] = (pa[((lit_u32(15u)) & 63u)] + f15);
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + lid.y) + (16u * k)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + lid.y) + (16u * k))) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + lid.x) + (16u * k2)) < cfg.d)) {
    h[((((wid.y * 64u) + lid.y) + (k * 16u)) * cfg.d) + (((wid.x * 64u) + lid.x) + (k2 * 16u))] = h[((((wid.y * 64u) + lid.y) + (k * 16u)) * cfg.d) + (((wid.x * 64u) + lid.x) + (k2 * 16u))] + pa[(((k * 4u) + k2) & 63u)];
    }
    }
    }
    }
}
