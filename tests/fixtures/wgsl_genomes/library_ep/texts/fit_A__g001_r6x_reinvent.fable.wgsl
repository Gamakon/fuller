// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var f16r: f32 = 0.0; var f17: f32 = 0.0; var f18: f32 = 0.0; var f19: f32 = 0.0; var f20: f32 = 0.0; var f21: f32 = 0.0; var f22: f32 = 0.0; var f23: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>;
    i0 = wid.x * lit_u32(128u);
    i1 = wid.y * lit_u32(64u);
    i2 = i0 + lid.x;
    i3 = i1 + lid.y;
    for (k = 0u; k < 32u; k = k + 1u) {
    {
    pa[((k) & 63u)] = lit_f32(0.0f);
    pb[((k) & 63u)] = lit_f32(0.0f);
    }
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    for (k = 0u; k < 8u; k = k + 1u) {
    {
    i4 = lidx + (lit_u32(256u) * k);
    i5 = t + (i4 >> 7u);
    i6 = i0 + (i4 & lit_u32(127u));
    f0 = lit_f32(0.0f);
    if ((i5 < cfg.m) && (i6 < cfg.d)) {
    f0 = phi[(i5 * cfg.d) + i6];
    }
    wg_a[(i4 & 2047u)] = f0;
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    {
    i4 = lidx + (lit_u32(256u) * k);
    i5 = t + (i4 >> 6u);
    i7 = i1 + (i4 & lit_u32(63u));
    f1 = lit_f32(0.0f);
    if ((i5 < cfg.m) && (i7 < cfg.r_rows) && ((cfg.r_off + i7) < cfg.d)) {
    f1 = phi[(i5 * cfg.d) + (cfg.r_off + i7)];
    }
    wg_b[(i4 & 2047u)] = f1;
    }
    }
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 2u) {
    {
    i4 = (k * lit_u32(128u)) + lid.x;
    i5 = (k * lit_u32(64u)) + lid.y;
    f0 = wg_a[((i4) & 2047u)];
    f1 = wg_a[((i4 + 16u) & 2047u)];
    f2 = wg_a[((i4 + 32u) & 2047u)];
    f3 = wg_a[((i4 + 48u) & 2047u)];
    f4 = wg_a[((i4 + 64u) & 2047u)];
    f5 = wg_a[((i4 + 80u) & 2047u)];
    f6 = wg_a[((i4 + 96u) & 2047u)];
    f7 = wg_a[((i4 + 112u) & 2047u)];
    f8 = wg_a[((i4 + 128u) & 2047u)];
    f9 = wg_a[((i4 + 144u) & 2047u)];
    f10 = wg_a[((i4 + 160u) & 2047u)];
    f11 = wg_a[((i4 + 176u) & 2047u)];
    f12 = wg_a[((i4 + 192u) & 2047u)];
    f13 = wg_a[((i4 + 208u) & 2047u)];
    f14 = wg_a[((i4 + 224u) & 2047u)];
    f15 = wg_a[((i4 + 240u) & 2047u)];
    f16r = wg_b[((i5) & 2047u)];
    f17 = wg_b[((i5 + 16u) & 2047u)];
    f18 = wg_b[((i5 + 32u) & 2047u)];
    f19 = wg_b[((i5 + 48u) & 2047u)];
    f20 = wg_b[((i5 + 64u) & 2047u)];
    f21 = wg_b[((i5 + 80u) & 2047u)];
    f22 = wg_b[((i5 + 96u) & 2047u)];
    f23 = wg_b[((i5 + 112u) & 2047u)];
    pa[0u] = pa[0u] + (f16r * f0); pa[1u] = pa[1u] + (f16r * f1); pa[2u] = pa[2u] + (f16r * f2); pa[3u] = pa[3u] + (f16r * f3);
    pa[4u] = pa[4u] + (f16r * f4); pa[5u] = pa[5u] + (f16r * f5); pa[6u] = pa[6u] + (f16r * f6); pa[7u] = pa[7u] + (f16r * f7);
    pa[8u] = pa[8u] + (f17 * f0); pa[9u] = pa[9u] + (f17 * f1); pa[10u] = pa[10u] + (f17 * f2); pa[11u] = pa[11u] + (f17 * f3);
    pa[12u] = pa[12u] + (f17 * f4); pa[13u] = pa[13u] + (f17 * f5); pa[14u] = pa[14u] + (f17 * f6); pa[15u] = pa[15u] + (f17 * f7);
    pa[16u] = pa[16u] + (f18 * f0); pa[17u] = pa[17u] + (f18 * f1); pa[18u] = pa[18u] + (f18 * f2); pa[19u] = pa[19u] + (f18 * f3);
    pa[20u] = pa[20u] + (f18 * f4); pa[21u] = pa[21u] + (f18 * f5); pa[22u] = pa[22u] + (f18 * f6); pa[23u] = pa[23u] + (f18 * f7);
    pa[24u] = pa[24u] + (f19 * f0); pa[25u] = pa[25u] + (f19 * f1); pa[26u] = pa[26u] + (f19 * f2); pa[27u] = pa[27u] + (f19 * f3);
    pa[28u] = pa[28u] + (f19 * f4); pa[29u] = pa[29u] + (f19 * f5); pa[30u] = pa[30u] + (f19 * f6); pa[31u] = pa[31u] + (f19 * f7);
    pb[0u] = pb[0u] + (f20 * f8); pb[1u] = pb[1u] + (f20 * f9); pb[2u] = pb[2u] + (f20 * f10); pb[3u] = pb[3u] + (f20 * f11);
    pb[4u] = pb[4u] + (f20 * f12); pb[5u] = pb[5u] + (f20 * f13); pb[6u] = pb[6u] + (f20 * f14); pb[7u] = pb[7u] + (f20 * f15);
    pb[8u] = pb[8u] + (f21 * f8); pb[9u] = pb[9u] + (f21 * f9); pb[10u] = pb[10u] + (f21 * f10); pb[11u] = pb[11u] + (f21 * f11);
    pb[12u] = pb[12u] + (f21 * f12); pb[13u] = pb[13u] + (f21 * f13); pb[14u] = pb[14u] + (f21 * f14); pb[15u] = pb[15u] + (f21 * f15);
    pb[16u] = pb[16u] + (f22 * f8); pb[17u] = pb[17u] + (f22 * f9); pb[18u] = pb[18u] + (f22 * f10); pb[19u] = pb[19u] + (f22 * f11);
    pb[20u] = pb[20u] + (f22 * f12); pb[21u] = pb[21u] + (f22 * f13); pb[22u] = pb[22u] + (f22 * f14); pb[23u] = pb[23u] + (f22 * f15);
    pb[24u] = pb[24u] + (f23 * f8); pb[25u] = pb[25u] + (f23 * f9); pb[26u] = pb[26u] + (f23 * f10); pb[27u] = pb[27u] + (f23 * f11);
    pb[28u] = pb[28u] + (f23 * f12); pb[29u] = pb[29u] + (f23 * f13); pb[30u] = pb[30u] + (f23 * f14); pb[31u] = pb[31u] + (f23 * f15);
    }
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    {
    i6 = i3 + (lit_u32(16u) * k);
    if ((i6 < cfg.r_rows) && ((cfg.r_off + i6) < cfg.d)) {
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    {
    i7 = i2 + (lit_u32(16u) * k2);
    if (i7 < cfg.d) {
    h[(i6 * cfg.d) + i7] = h[(i6 * cfg.d) + i7] + (pa[(((k * lit_u32(8u)) + k2)) & 63u] + pb[(((k * lit_u32(8u)) + k2)) & 63u]);
    }
    }
    }
    }
    }
    }
}
