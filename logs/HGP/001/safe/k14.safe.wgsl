// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0;
    var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u;
    var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; var i15: u32 = 0u;
    var b0: bool = false; var b1: bool = false; var b2: bool = false; var b3: bool = false;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>; var pd: array<f32, 64>; var pe: array<f32, 64>; var pf: array<f32, 64>;
    var pv: array<vec4<f32>, 16>; var pi: array<u32, 16>; var pj: array<u32, 16>; var pk: array<u32, 16>; var pm: array<bool, 16>; var pn: array<bool, 16>;
    v0 = vec4<f32>(lit_f32(0.0f));
    v1 = vec4<f32>(lit_f32(0.0f));
    v2 = vec4<f32>(lit_f32(0.0f));
    {
    v3 = vec4<f32>(lit_f32(0.0f));
    i0 = lit_u32(0u);
    i1 = lit_u32(0u);
    {
    for (t = 0u; t < cfg.m; t = t + 16u) {
    {
    v4 = vec4<f32>(lit_f32(0.0f));
    if (((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(16u))) < cfg.m)) {
    if ((((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u))) < cfg.d)) {
    pa[((lit_u32(0u)) & 63u)] = phi[(((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(16u))) * cfg.d) + ((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u))))];
    }
    if (((((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u))) + lit_u32(1u)) < cfg.d)) {
    pa[((lit_u32(1u)) & 63u)] = phi[((((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(16u))) * cfg.d) + ((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u)))) + lit_u32(1u))];
    }
    if (((((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u))) + lit_u32(2u)) < cfg.d)) {
    pa[((lit_u32(2u)) & 63u)] = phi[((((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(16u))) * cfg.d) + ((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u)))) + lit_u32(2u))];
    }
    if (((((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u))) + lit_u32(3u)) < cfg.d)) {
    pa[((lit_u32(3u)) & 63u)] = phi[((((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(16u))) * cfg.d) + ((wid.x * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) % lit_u32(16u)) * lit_u32(4u)))) + lit_u32(3u))];
    }
    }
    wg_va[((((lid.y * lit_u32(16u)) + lid.x)) & 255u)] = v4;
    }
    i2 = lit_u32(0u);
    for (k = 0u; k < 4u; k = k + 1u) {
    {
    f0 = lit_f32(0.0f);
    if (((((t + ((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u))) / lit_u32(64u))) < cfg.m) && (((wid.y * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u))) % lit_u32(64u))) < cfg.r_rows)) && ((cfg.r_off + ((wid.y * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u))) % lit_u32(64u)))) < cfg.d))) {
    f0 = phi[(((t + ((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u))) / lit_u32(64u))) * cfg.d) + (cfg.r_off + ((wid.y * lit_u32(64u)) + ((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u))) % lit_u32(64u)))))];
    }
    wg_a[(((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & 2047u)] = f0;
    }
    }
    {
    workgroupBarrier();
    i3 = lit_u32(0u);
    for (k = 0u; k < 16u; k = k + 1u) {
    {
    v0 = (v0 + (vec4<f32>(wg_a[((((k * lit_u32(64u)) + (lid.y * lit_u32(4u)))) & 2047u)]) * wg_va[((((k * lit_u32(16u)) + lid.x)) & 255u)]));
    v1 = (v1 + (vec4<f32>(wg_a[(((((k * lit_u32(64u)) + (lid.y * lit_u32(4u))) + lit_u32(1u))) & 2047u)]) * wg_va[((((k * lit_u32(16u)) + lid.x)) & 255u)]));
    v2 = (v2 + (vec4<f32>(wg_a[(((((k * lit_u32(64u)) + (lid.y * lit_u32(4u))) + lit_u32(2u))) & 2047u)]) * wg_va[((((k * lit_u32(16u)) + lid.x)) & 255u)]));
    v3 = (v3 + (vec4<f32>(wg_a[(((((k * lit_u32(64u)) + (lid.y * lit_u32(4u))) + lit_u32(3u))) & 2047u)]) * wg_va[((((k * lit_u32(16u)) + lid.x)) & 255u)]));
    }
    }
    workgroupBarrier();
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    {
    if (((((wid.y * lit_u32(64u)) + (lid.y * lit_u32(4u))) + k) < cfg.r_rows)) {
    v5 = v0;
    if ((k == lit_u32(1u))) {
    v5 = v1;
    }
    if ((k == lit_u32(2u))) {
    v5 = v2;
    }
    {
    if ((k == lit_u32(3u))) {
    v5 = v3;
    }
    if ((((wid.x * lit_u32(64u)) + (lid.x * lit_u32(4u))) < cfg.d)) {
    h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] + pb[((lit_u32(0u)) & 63u)];
    }
    if (((((wid.x * lit_u32(64u)) + (lid.x * lit_u32(4u))) + lit_u32(1u)) < cfg.d)) {
    h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] + pb[((lit_u32(1u)) & 63u)];
    }
    {
    if (((((wid.x * lit_u32(64u)) + (lid.x * lit_u32(4u))) + lit_u32(2u)) < cfg.d)) {
    h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] + pb[((lit_u32(2u)) & 63u)];
    }
    if (((((wid.x * lit_u32(64u)) + (lid.x * lit_u32(4u))) + lit_u32(3u)) < cfg.d)) {
    h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] + pb[((lit_u32(3u)) & 63u)];
    }
    }
    }
    }
    }
    }
    }
    }
}
