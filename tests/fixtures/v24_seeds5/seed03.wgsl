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
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_au_a: array<atomic<u32>, 513>;
var<workgroup> wg_au_b: array<atomic<u32>, 1>;
var<workgroup> wg_ua: array<u32, 16>;
var<workgroup> wg_ub: array<u32, 1>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 1u) {
    f0 = keys[(((((wid.x) & (8191u)) * 2048u) + t))];
    i1 = 0u;
    i1 = (((t * 16u) + ((lid.x) >> ((5u) & 31u))) + 1u);
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i0 = i1;
    i1 = 0u;
    i1 = (((t * 16u) + (((lid.x) >> ((5u) & 31u)) + 8u)) + 1u);
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i2 = i1;
    if ((((((i0) >> ((lid.x) & 31u))) & (1u)) == 1u)) {
    f1 = (f1 + f0);
    } else {
    f1 = (f1 - f0);
    }
    if ((((((i2) >> ((lid.x) & 31u))) & (1u)) == 1u)) {
    f2 = (f2 + f0);
    } else {
    f2 = (f2 - f0);
    }
    }
    wg_a[((lid.x) & 511u)] = f1;
    wg_a[(((lid.x + 256u)) & 511u)] = f2;
    workgroupBarrier();
    if ((lid.x < 16u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    if ((wg_a[((((lid.x * 32u) + k)) & 511u)] > 0.0f)) {
    i3 = ((i3) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    index_u32[((((((((cfg.arrived_begin + wid.x)) & (32767u)) * 16u) + lid.x)) & (8388607u)))] = (i3);
    }
}
@compute @workgroup_size(256, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
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
    var i12: u32 = 0u;
    var i13: u32 = 0u;
    var i14: u32 = 0u;
    var b0: bool = false;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    i0 = 512u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 1u) {
    f0 = queries[(((((wid.x) & (65535u)) * 2048u) + t))];
    i2 = 0u;
    i2 = (((t * 16u) + ((lid.x) >> ((5u) & 31u))) + 1u);
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i1 = i2;
    i2 = 0u;
    i2 = (((t * 16u) + (((lid.x) >> ((5u) & 31u)) + 8u)) + 1u);
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i3 = i2;
    if ((((((i1) >> ((lid.x) & 31u))) & (1u)) == 1u)) {
    f1 = (f1 + f0);
    } else {
    f1 = (f1 - f0);
    }
    if ((((((i3) >> ((lid.x) & 31u))) & (1u)) == 1u)) {
    f2 = (f2 + f0);
    } else {
    f2 = (f2 - f0);
    }
    }
    wg_a[((lid.x) & 511u)] = f1;
    wg_a[(((lid.x + 256u)) & 511u)] = f2;
    for (t = 0u; t < min((lit_u32(513u)), 16777216u); t = t + 256u) {
    if (((t + lid.x) < 513u)) {
    atomicStore(&wg_au_a[min(((t + lid.x)), 512u)], 0u);
    }
    }
    if ((lid.x == 0u)) {
    atomicStore(&wg_au_b[0u], 0u);
    }
    workgroupBarrier();
    if ((lid.x < 16u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    if ((wg_a[((((lid.x * 32u) + k)) & 511u)] > 0.0f)) {
    i4 = ((i4) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    wg_ua[((lid.x) & 15u)] = i4;
    }
    workgroupBarrier();
    for (k = 0u; k < min((lit_u32(25088u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    i5 = 0u;
    i6 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(16u)), 16777216u); k2 = k2 + 1u) {
    i5 = (i5 + countOneBits((u32(index_u32[((((((((k + lid.x)) & (32767u)) * 16u) + k2)) & (8388607u)))]) ^ u32(wg_ua[((k2) & 15u)]))));
    }
    _ = atomicAdd(&wg_au_a[min((i5), 512u)], 1u);
    }
    }
    workgroupBarrier();
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(513u)), 16777216u); k = k + 1u) {
    b0 = false;
    i7 = (i7 + atomicLoad(&wg_au_a[min((k), 512u)]));
    if ((i7 >= 64u)) {
    i0 = k;
    b0 = true;
    }
    if (b0) { break; }
    }
    wg_ub[0u] = i0;
    }
    workgroupBarrier();
    i8 = wg_ub[0u];
    for (k = 0u; k < min((lit_u32(25088u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    i9 = 0u;
    i10 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(16u)), 16777216u); k2 = k2 + 1u) {
    i9 = (i9 + countOneBits((u32(index_u32[((((((((k + lid.x)) & (32767u)) * 16u) + k2)) & (8388607u)))]) ^ u32(wg_ua[((k2) & 15u)]))));
    }
    if ((i9 < i8)) {
    i11 = atomicAdd(&wg_au_b[0u], 1u);
    if (((i11 < 64u) & (wid.x < cfg.n_queries))) {
    cands[(((wid.x * 64u) + i11))] = ((k + lid.x));
    }
    }
    }
    }
    workgroupBarrier();
    for (k = 0u; k < min((lit_u32(25088u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    i12 = 0u;
    i13 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(16u)), 16777216u); k2 = k2 + 1u) {
    i12 = (i12 + countOneBits((u32(index_u32[((((((((k + lid.x)) & (32767u)) * 16u) + k2)) & (8388607u)))]) ^ u32(wg_ua[((k2) & 15u)]))));
    }
    if ((i12 == i8)) {
    i14 = atomicAdd(&wg_au_b[0u], 1u);
    if (((i14 < 64u) & (wid.x < cfg.n_queries))) {
    cands[(((wid.x * 64u) + i14))] = ((k + lid.x));
    }
    }
    }
    }
}
