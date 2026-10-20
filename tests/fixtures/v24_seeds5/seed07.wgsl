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
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_au_a: array<atomic<u32>, 257>;
var<workgroup> wg_au_b: array<atomic<u32>, 1>;
var<workgroup> wg_ua: array<u32, 256>;
var<workgroup> wg_ub: array<u32, 8>;
var<workgroup> wg_uc: array<u32, 1>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 256u) {
    wg_a[(((t + lid.x)) & 2047u)] = keys[((((((wid.x) & (8191u)) * 2048u) + t) + lid.x))];
    }
    workgroupBarrier();
    f1 = 0.0f;
    i0 = 0u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 1u) {
    i2 = 0u;
    i3 = (((t * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + (((lid.x) >> ((5u) & 31u)) * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))))) + ((u32((lit_u32(291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(17767u))));
    i3 = u32((u32(i3) ^ (u32(i3) >> ((16u) & 31u))));
    i3 = (i3 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i3 = u32((u32(i3) ^ (u32(i3) >> ((15u) & 31u))));
    i3 = (i3 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i3 = u32((u32(i3) ^ (u32(i3) >> ((16u) & 31u))));
    i1 = i3;
    f1 = (f1 + select((-wg_a[((t) & 2047u)]), wg_a[((t) & 2047u)], (((((i1) >> ((lid.x) & 31u))) & (1u)) == 1u)));
    }
    f0 = f1;
    wg_ua[((lid.x) & 255u)] = select(0u, 1u, (f0 >= 0.0f));
    workgroupBarrier();
    if ((lid.x < 8u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    i4 = ((i4) | (u32((u32(wg_ua[((((lid.x * 32u) + k)) & 255u)]) << ((u32(k)) & 31u)))));
    }
    if ((!((cfg.arrived_begin + wid.x) >= cfg.arrived_end))) {
    index_u32[((((((cfg.arrived_begin + wid.x)) & (65535u)) * 8u) + lid.x))] = (i4);
    }
    }
}
@compute @workgroup_size(256, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
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
    var i15: u32 = 0u;
    var b0: bool = false;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var pi: array<u32, 16>;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 256u) {
    wg_a[(((t + lid.x)) & 2047u)] = queries[((((((wid.x) & (4095u)) * 2048u) + t) + lid.x))];
    }
    workgroupBarrier();
    f1 = 0.0f;
    i0 = 0u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 1u) {
    i2 = 0u;
    i3 = (((t * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + (((lid.x) >> ((5u) & 31u)) * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))))) + ((u32((lit_u32(291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(17767u))));
    i3 = u32((u32(i3) ^ (u32(i3) >> ((16u) & 31u))));
    i3 = (i3 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i3 = u32((u32(i3) ^ (u32(i3) >> ((15u) & 31u))));
    i3 = (i3 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i3 = u32((u32(i3) ^ (u32(i3) >> ((16u) & 31u))));
    i1 = i3;
    f1 = (f1 + select((-wg_a[((t) & 2047u)]), wg_a[((t) & 2047u)], (((((i1) >> ((lid.x) & 31u))) & (1u)) == 1u)));
    }
    f0 = f1;
    wg_ua[((lid.x) & 255u)] = select(0u, 1u, (f0 >= 0.0f));
    atomicStore(&wg_au_a[min((lid.x), 256u)], 0u);
    if ((lid.x == 0u)) {
    atomicStore(&wg_au_a[256u], 0u);
    atomicStore(&wg_au_b[0u], 0u);
    }
    workgroupBarrier();
    if ((lid.x < 8u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    i4 = ((i4) | (u32((u32(wg_ua[((((lid.x * 32u) + k)) & 255u)]) << ((u32(k)) & 31u)))));
    }
    wg_ub[((lid.x) & 7u)] = i4;
    }
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    pi[((k) & 15u)] = wg_ub[((k) & 7u)];
    }
    for (t = 0u; t < min((cfg.arrived_end), 16777216u); t = t + 256u) {
    if (((t + lid.x) < cfg.arrived_end)) {
    i5 = 0u;
    i6 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i5 = (i5 + countOneBits((u32(index_u32[((((((t + lid.x)) & (65535u)) * 8u) + k))]) ^ u32(pi[((k) & 15u)]))));
    }
    _ = atomicAdd(&wg_au_a[min((i5), 256u)], 1u);
    }
    }
    workgroupBarrier();
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(257u)), 16777216u); k = k + 1u) {
    b0 = false;
    i7 = (i7 + atomicLoad(&wg_au_a[min((k), 256u)]));
    if ((i7 >= 64u)) {
    i8 = k;
    b0 = true;
    }
    if (b0) { break; }
    }
    wg_uc[0u] = i8;
    }
    workgroupBarrier();
    i9 = wg_uc[0u];
    for (t = 0u; t < min((cfg.arrived_end), 16777216u); t = t + 256u) {
    if (((t + lid.x) < cfg.arrived_end)) {
    i10 = 0u;
    i11 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i10 = (i10 + countOneBits((u32(index_u32[((((((t + lid.x)) & (65535u)) * 8u) + k))]) ^ u32(pi[((k) & 15u)]))));
    }
    if ((i10 < i9)) {
    i12 = atomicAdd(&wg_au_b[0u], 1u);
    if ((i12 < 64u)) {
    if ((!(wid.x >= cfg.n_queries))) {
    cands[(((((wid.x) & (4095u)) * 64u) + i12))] = ((t + lid.x));
    }
    }
    }
    }
    }
    workgroupBarrier();
    for (t = 0u; t < min((cfg.arrived_end), 16777216u); t = t + 256u) {
    if (((t + lid.x) < cfg.arrived_end)) {
    i13 = 0u;
    i14 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    i13 = (i13 + countOneBits((u32(index_u32[((((((t + lid.x)) & (65535u)) * 8u) + k))]) ^ u32(pi[((k) & 15u)]))));
    }
    if ((i13 == i9)) {
    i15 = atomicAdd(&wg_au_b[0u], 1u);
    if ((i15 < 64u)) {
    if ((!(wid.x >= cfg.n_queries))) {
    cands[(((((wid.x) & (4095u)) * 64u) + i15))] = ((t + lid.x));
    }
    }
    }
    }
    }
    return;
}
