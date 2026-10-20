// grid 256 1 1 1
// entry ingest grid 256 1 1 1
// entry search grid 256 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
var<workgroup> wg_c: array<f32, 1024>;
var<workgroup> wg_ua: array<u32, 1024>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var k: u32 = 0u;
    i0 = 128u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i1 = 0u;
    i1 = (((lid.x + (256u * k)) + 12345u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    if ((((i1) & (1u)) == 0u)) {
    f0 = 1.0f;
    } else {
    f0 = -1.0f;
    }
    f1 = (f1 + (f0 * keys[(((((((wid.x) & (8191u)) * 2048u) + (lid.x + (256u * k)))) & (16777215u)))]));
    }
    wg_a[((lid.x) & 255u)] = (f1 * f1);
    workgroupBarrier();
    workgroupBarrier();
    if (lidx < 128u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 128u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 64u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 64u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 32u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 32u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 16u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 16u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 8u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 8u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 4u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 4u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 2u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 2u) & 255u)]); }
    workgroupBarrier();
    if (lidx < 1u) { wg_a[((lidx) & 255u)] = (wg_a[((lidx) & 255u)] + wg_a[((lidx + 1u) & 255u)]); }
    workgroupBarrier();
    f2 = wg_a[0u];
    if ((f2 > 0.0f)) {
    f3 = (f1 * inverseSqrt(f2));
    }
    index_f32[((((((((cfg.arrived_begin + wid.x)) & (32767u)) * 256u) + lid.x)) & (8388607u)))] = (f3);
}
@compute @workgroup_size(256, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var f6: f32 = 0.0;
    var f7: f32 = 0.0;
    var f8: f32 = 0.0;
    var f9: f32 = 0.0;
    var f10: f32 = 0.0;
    var f11: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var i7: u32 = 0u;
    var i8: u32 = 0u;
    var i9: u32 = 0u;
    var i10: u32 = 0u;
    var i11: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    f0 = -3e38f;
    f1 = -3e38f;
    f2 = -3e38f;
    f3 = -3e38f;
    i0 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    i1 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    i2 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    i3 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    for (k = 0u; k < 8u; k = k + 1u) {
    i4 = 0u;
    i4 = (((lid.x + (256u * k)) + 12345u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((15u) & 31u))));
    i4 = (i4 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((13u) & 31u))));
    if ((((i4) & (1u)) == 0u)) {
    f4 = 1.0f;
    } else {
    f4 = -1.0f;
    }
    f5 = (f5 + (f4 * queries[(((((((wid.x) & (4095u)) * 2048u) + (lid.x + (256u * k)))) & (16777215u)))]));
    }
    wg_b[((lid.x) & 255u)] = f5;
    workgroupBarrier();
    for (t = 0u; t < min((cfg.arrived_end), 16777216u); t = t + 256u) {
    if (((t + lid.x) < cfg.arrived_end)) {
    f6 = 0.0f;
    i5 = 0u;
    for (k = 0u; k < min((lit_u32(256u)), 16777216u); k = k + 1u) {
    f6 = (f6 + (index_f32[((((((((t + lid.x)) & (32767u)) * 256u) + k)) & (8388607u)))] * wg_b[((k) & 255u)]));
    }
    f7 = f6;
    i6 = (t + lid.x);
    if ((f7 > f0)) {
    f8 = f0;
    i7 = i0;
    f0 = f7;
    i0 = i6;
    f7 = f8;
    i6 = i7;
    }
    if ((f7 > f1)) {
    f9 = f1;
    i8 = i1;
    f1 = f7;
    i1 = i6;
    f7 = f9;
    i6 = i8;
    }
    if ((f7 > f2)) {
    f10 = f2;
    i9 = i2;
    f2 = f7;
    i2 = i6;
    f7 = f10;
    i6 = i9;
    }
    if ((f7 > f3)) {
    f3 = f7;
    i3 = i6;
    }
    }
    }
    wg_c[(((lid.x * 4u)) & 1023u)] = f0;
    wg_ua[(((lid.x * 4u)) & 1023u)] = i0;
    wg_c[((((lid.x * 4u) + 1u)) & 1023u)] = f1;
    wg_ua[((((lid.x * 4u) + 1u)) & 1023u)] = i1;
    wg_c[((((lid.x * 4u) + 2u)) & 1023u)] = f2;
    wg_ua[((((lid.x * 4u) + 2u)) & 1023u)] = i2;
    wg_c[((((lid.x * 4u) + 3u)) & 1023u)] = f3;
    wg_ua[((((lid.x * 4u) + 3u)) & 1023u)] = i3;
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    f11 = wg_c[((((lid.x * 4u) + k)) & 1023u)];
    if ((wg_ua[((((lid.x * 4u) + k)) & 1023u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    i10 = 0u;
    i11 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(1024u)), 16777216u); k2 = k2 + 1u) {
    if (((wg_ua[((k2) & 1023u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & ((wg_c[((k2) & 1023u)] > f11) | ((wg_c[((k2) & 1023u)] == f11) & (k2 < ((lid.x * 4u) + k)))))) {
    i10 = (i10 + 1u);
    }
    }
    if ((i10 < 64u)) {
    cands[(((wid.x * 64u) + i10))] = (wg_ua[((((lid.x * 4u) + k)) & 1023u)]);
    }
    }
    }
    }
}
