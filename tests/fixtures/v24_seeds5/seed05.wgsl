// grid 64 1 1 1
// entry ingest grid 64 1 1 1
// entry search grid 64 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_b: array<f32, 64>;
var<workgroup> wg_c: array<f32, 256>;
var<workgroup> wg_ua: array<u32, 256>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var f6: f32 = 0.0;
    var f7: f32 = 0.0;
    var f8: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var k: u32 = 0u;
    i0 = 32u;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    i1 = 0u;
    i2 = ((((k * 4u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i2) & (1u)) == 1u));
    i1 = 0u;
    i2 = (((((k * 4u) + 1u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    f2 = select(-1.0f, 1.0f, (((i2) & (1u)) == 1u));
    i1 = 0u;
    i2 = (((((k * 4u) + 2u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    f3 = select(-1.0f, 1.0f, (((i2) & (1u)) == 1u));
    i1 = 0u;
    i2 = (((((k * 4u) + 3u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    f4 = select(-1.0f, 1.0f, (((i2) & (1u)) == 1u));
    f0 = ((((keys[(((((wid.x) & (8191u)) * 512u) + k))].x * f1) + (keys[(((((wid.x) & (8191u)) * 512u) + k))].y * f2)) + (keys[(((((wid.x) & (8191u)) * 512u) + k))].z * f3)) + (keys[(((((wid.x) & (8191u)) * 512u) + k))].w * f4));
    f5 = (f5 + f0);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    f6 = (f6 + dot(keys[((((((wid.x) & (8191u)) * 512u) + (k * 64u)) + lid.x))], keys[((((((wid.x) & (8191u)) * 512u) + (k * 64u)) + lid.x))]));
    }
    }
    wg_a[((lid.x) & 63u)] = f6;
    workgroupBarrier();
    workgroupBarrier();
    if (lidx < 32u) { wg_a[((lidx) & 63u)] = (wg_a[((lidx) & 63u)] + wg_a[((lidx + 32u) & 63u)]); }
    workgroupBarrier();
    if (lidx < 16u) { wg_a[((lidx) & 63u)] = (wg_a[((lidx) & 63u)] + wg_a[((lidx + 16u) & 63u)]); }
    workgroupBarrier();
    if (lidx < 8u) { wg_a[((lidx) & 63u)] = (wg_a[((lidx) & 63u)] + wg_a[((lidx + 8u) & 63u)]); }
    workgroupBarrier();
    if (lidx < 4u) { wg_a[((lidx) & 63u)] = (wg_a[((lidx) & 63u)] + wg_a[((lidx + 4u) & 63u)]); }
    workgroupBarrier();
    if (lidx < 2u) { wg_a[((lidx) & 63u)] = (wg_a[((lidx) & 63u)] + wg_a[((lidx + 2u) & 63u)]); }
    workgroupBarrier();
    if (lidx < 1u) { wg_a[((lidx) & 63u)] = (wg_a[((lidx) & 63u)] + wg_a[((lidx + 1u) & 63u)]); }
    workgroupBarrier();
    f7 = wg_a[0u];
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    if ((f7 > 0.0f)) {
    f8 = (f5 * inverseSqrt(f7));
    }
    index_f32[((((((((cfg.arrived_begin + wid.x)) & (524287u)) * 64u) + lid.x)) & (8388607u)))] = (f8);
    }
}
@compute @workgroup_size(64, 1, 1)
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
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var i7: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    pa[0u] = -3e38f;
    pa[1u] = -3e38f;
    pa[2u] = -3e38f;
    pa[3u] = -3e38f;
    pi[0u] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    pi[1u] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    pi[2u] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    pi[3u] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    i0 = 0u;
    i1 = ((((k * 4u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((k * 4u) + 1u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f2 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((k * 4u) + 2u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f3 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((k * 4u) + 3u) * 64u) + lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51831u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44605u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f4 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    f0 = ((((queries[(((((wid.x) & (65535u)) * 512u) + k))].x * f1) + (queries[(((((wid.x) & (65535u)) * 512u) + k))].y * f2)) + (queries[(((((wid.x) & (65535u)) * 512u) + k))].z * f3)) + (queries[(((((wid.x) & (65535u)) * 512u) + k))].w * f4));
    f5 = (f5 + f0);
    }
    }
    wg_b[((lid.x) & 63u)] = f5;
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((cfg.arrived_end), 16777216u); k = k + 64u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    f6 = 0.0f;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    f6 = (f6 + (wg_b[((k2) & 63u)] * index_f32[((((((((k + lid.x)) & (524287u)) * 64u) + k2)) & (8388607u)))]));
    }
    if ((f6 > pa[3u])) {
    f7 = f6;
    i2 = (k + lid.x);
    i3 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((f7 > pa[((k2) & 63u)])) {
    f8 = pa[((k2) & 63u)];
    i4 = pi[((k2) & 15u)];
    pa[((k2) & 63u)] = f7;
    pi[((k2) & 15u)] = i2;
    f7 = f8;
    i2 = i4;
    }
    }
    }
    }
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    wg_c[((((lid.x * 4u) + k)) & 255u)] = pa[((k) & 63u)];
    wg_ua[((((lid.x * 4u) + k)) & 255u)] = pi[((k) & 15u)];
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i5 = wg_ua[((((lid.x * 4u) + k)) & 255u)];
    if ((i5 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    f9 = wg_c[((((lid.x * 4u) + k)) & 255u)];
    i6 = 0u;
    i7 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(256u)), 16777216u); k2 = k2 + 1u) {
    if ((wg_ua[((k2) & 255u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    if (((wg_c[((k2) & 255u)] > f9) | ((wg_c[((k2) & 255u)] == f9) & (k2 < ((lid.x * 4u) + k))))) {
    i6 = (i6 + 1u);
    }
    }
    }
    if ((i6 < 64u)) {
    cands[(((wid.x * 64u) + i6))] = (i5);
    }
    }
    }
    }
}
