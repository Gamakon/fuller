// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pa: array<f32, 64>;
    i0 = lit_u32(0u);
    i1 = lit_u32(0u);
    i2 = lit_u32(0u);
    {
    for (k = 0u; k < 16u; k = k + 1u) {
    {
    pa[((k) & 63u)] = lit_f32(0.0f);
    }
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    {
    i3 = lit_u32(0u);
    for (k = 0u; k < 4u; k = k + 1u) {
    {
    f0 = lit_f32(0.0f);
    f1 = lit_f32(0.0f);
    if (((t + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) >> ((lit_u32(6u)) & 31u))) < cfg.m)) {
    if (((((wid.y * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u))))) < cfg.d))) {
    f0 = phi[(((t + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) >> ((lit_u32(6u)) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u))))))];
    }
    if ((((wid.x * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))) < cfg.d)) {
    f1 = phi[(((t + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) >> ((lit_u32(6u)) & 31u))) * cfg.d) + ((wid.x * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))))];
    }
    }
    {
    wg_a[(((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & 2047u)] = f0;
    wg_b[(((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & 2047u)] = f1;
    }
    }
    }
    workgroupBarrier();
    {
    i4 = lit_u32(0u);
    for (k = 0u; k < 16u; k = k + 1u) {
    {
    pa[((lit_u32(0u)) & 63u)] = (pa[((lit_u32(0u)) & 63u)] + (wg_a[((((k * lit_u32(64u)) + lid.y)) & 2047u)] * wg_b[((((k * lit_u32(64u)) + lid.x)) & 2047u)]));
    pa[((lit_u32(1u)) & 63u)] = (pa[((lit_u32(1u)) & 63u)] + (wg_a[((((k * lit_u32(64u)) + lid.y)) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    pa[((lit_u32(2u)) & 63u)] = (pa[((lit_u32(2u)) & 63u)] + (wg_a[((((k * lit_u32(64u)) + lid.y)) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(32u))) & 2047u)]));
    pa[((lit_u32(3u)) & 63u)] = (pa[((lit_u32(3u)) & 63u)] + (wg_a[((((k * lit_u32(64u)) + lid.y)) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(48u))) & 2047u)]));
    pa[((lit_u32(4u)) & 63u)] = (pa[((lit_u32(4u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[((((k * lit_u32(64u)) + lid.x)) & 2047u)]));
    pa[((lit_u32(5u)) & 63u)] = (pa[((lit_u32(5u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    pa[((lit_u32(6u)) & 63u)] = (pa[((lit_u32(6u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(32u))) & 2047u)]));
    pa[((lit_u32(7u)) & 63u)] = (pa[((lit_u32(7u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(48u))) & 2047u)]));
    pa[((lit_u32(8u)) & 63u)] = (pa[((lit_u32(8u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(32u))) & 2047u)] * wg_b[((((k * lit_u32(64u)) + lid.x)) & 2047u)]));
    pa[((lit_u32(9u)) & 63u)] = (pa[((lit_u32(9u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(32u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    pa[((lit_u32(10u)) & 63u)] = (pa[((lit_u32(10u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(32u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(32u))) & 2047u)]));
    pa[((lit_u32(11u)) & 63u)] = (pa[((lit_u32(11u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(32u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(48u))) & 2047u)]));
    pa[((lit_u32(12u)) & 63u)] = (pa[((lit_u32(12u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(48u))) & 2047u)] * wg_b[((((k * lit_u32(64u)) + lid.x)) & 2047u)]));
    pa[((lit_u32(13u)) & 63u)] = (pa[((lit_u32(13u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(48u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    pa[((lit_u32(14u)) & 63u)] = (pa[((lit_u32(14u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(48u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(32u))) & 2047u)]));
    pa[((lit_u32(15u)) & 63u)] = (pa[((lit_u32(15u)) & 63u)] + (wg_a[(((((k * lit_u32(64u)) + lid.y) + lit_u32(48u))) & 2047u)] * wg_b[(((((k * lit_u32(64u)) + lid.x) + lit_u32(48u))) & 2047u)]));
    }
    }
    }
    workgroupBarrier();
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    {
    if ((((((wid.y * lit_u32(64u)) + lid.y) + (k * lit_u32(16u))) < cfg.r_rows) && ((cfg.r_off + (((wid.y * lit_u32(64u)) + lid.y) + (k * lit_u32(16u)))) < cfg.d))) {
    i5 = lit_u32(0u);
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    {
    if (((((wid.x * lit_u32(64u)) + lid.x) + (k2 * lit_u32(16u))) < cfg.d)) {
    if ((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows && (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d && (cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + pa[((((k * lit_u32(4u)) + k2)) & 63u)]; }
    }
    }
    }
    }
    }
    }
    }
}
