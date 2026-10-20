// grid 64 1 1 1
// entry ingest grid 64 1 1 1
// entry search grid 64 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_au_a: array<atomic<u32>, 2048>;
var<workgroup> wg_au_b: array<atomic<u32>, 1>;
var<workgroup> wg_au_c: array<atomic<u32>, 1>;
var<workgroup> wg_b: array<f32, 64>;
var<workgroup> wg_ua: array<u32, 1>;
var<workgroup> wg_ub: array<u32, 1>;
var<workgroup> wg_uc: array<u32, 4>;
var<workgroup> wg_ud: array<u32, 256>;
var<workgroup> wg_ue: array<u32, 1>;
var<workgroup> wg_uf: array<u32, 1>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var k: u32 = 0u;
    f0 = -1e30f;
    if ((((cfg.pass_index == 0u) & (cfg.stage == 1u)) & (wid.x < 64u))) {
    for (k = 0u; k < 32u; k = k + 1u) {
    f1 = keys[(((((wid.x) & (8191u)) * 2048u) + ((k * 64u) + lid.x)))];
    index_f32[(((((((wid.x) & (63u)) * 2048u) + ((k * 64u) + lid.x))) & (8388607u)))] = (f1);
    f2 = (f2 + (f1 * f1));
    }
    }
    wg_a[((lid.x) & 63u)] = f2;
    workgroupBarrier();
    if (((((cfg.pass_index == 0u) & (cfg.stage == 1u)) & (wid.x < 64u)) & (lid.x == 0u))) {
    for (k = 0u; k < 64u; k = k + 1u) {
    f3 = (f3 + wg_a[((k) & 63u)]);
    }
    index_f32[((131072u + ((wid.x) & (63u))))] = ((1.0f / sqrt(max(f3, 1e-20f))));
    }
    workgroupBarrier();
    if ((cfg.pass_index == 1u)) {
    for (k = 0u; k < min((lit_u32(2048u)), 16777216u); k = k + 1u) {
    f4 = (f4 + (keys[(((((wid.x) & (8191u)) * 2048u) + k))] * index_f32[(((((lid.x * 2048u) + k)) & (8388607u)))]));
    }
    f0 = (f4 * index_f32[((131072u + lid.x))]);
    }
    wg_a[((lid.x) & 63u)] = f0;
    workgroupBarrier();
    if ((lid.x == 0u)) {
    f5 = wg_a[0u];
    for (k = 0u; k < 63u; k = k + 1u) {
    if ((wg_a[(((1u + k)) & 63u)] > f5)) {
    f5 = wg_a[(((1u + k)) & 63u)];
    i0 = (1u + k);
    }
    }
    wg_ua[0u] = i0;
    }
    workgroupBarrier();
    i1 = ((wg_ua[0u]) & (63u));
    if ((cfg.pass_index == 1u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    if (((keys[(((((wid.x) & (8191u)) * 2048u) + ((lid.x * 32u) + k)))] - index_f32[(((((i1 * 2048u) + ((lid.x * 32u) + k))) & (8388607u)))]) > 0.0f)) {
    i2 = ((i2) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    }
    if (((cfg.pass_index == 1u) & (lid.x == 0u))) {
    wg_ub[0u] = atomicAdd(&index_u32[(i1)], (1u));
    }
    workgroupBarrier();
    if (((cfg.pass_index == 1u) & (wg_ub[0u] < 1792u))) {
    atomicStore(&index_u32[(((((114752u + (((i1 * 1792u) + wg_ub[0u]) * 64u)) + lid.x)) & (8388607u)))], (i2));
    if ((lid.x == 0u)) {
    atomicStore(&index_u32[(((((64u + (i1 * 1792u)) + wg_ub[0u])) & (8388607u)))], ((cfg.arrived_begin + wid.x)));
    }
    }
}
@compute @workgroup_size(64, 1, 1)
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
    var i16: u32 = 0u;
    var i17: u32 = 0u;
    var i18: u32 = 0u;
    var b0: bool = false;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    i0 = 2048u;
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(2048u)), 16777216u); k = k + 1u) {
    f0 = (f0 + (queries[(((((wid.x) & (65535u)) * 2048u) + k))] * index_f32[(((((lid.x * 2048u) + k)) & (8388607u)))]));
    }
    f0 = (f0 * index_f32[((131072u + lid.x))]);
    }
    if ((atomicLoad(&index_u32[(lid.x)]) == 0u)) {
    f0 = -1e30f;
    }
    wg_b[((lid.x) & 63u)] = f0;
    for (k = 0u; k < 32u; k = k + 1u) {
    atomicStore(&wg_au_a[((((k * 64u) + lid.x)) & 2047u)], 0u);
    }
    if ((lid.x == 0u)) {
    atomicStore(&wg_au_b[0u], 0u);
    atomicStore(&wg_au_c[0u], 0u);
    }
    workgroupBarrier();
    if ((lid.x == 0u)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i1 = 0u;
    f1 = wg_b[0u];
    for (k2 = 0u; k2 < 63u; k2 = k2 + 1u) {
    if ((wg_b[(((1u + k2)) & 63u)] > f1)) {
    f1 = wg_b[(((1u + k2)) & 63u)];
    i1 = (1u + k2);
    }
    }
    wg_uc[((k) & 3u)] = i1;
    wg_b[((i1) & 63u)] = -2e30f;
    }
    }
    workgroupBarrier();
    for (k = 0u; k < 4u; k = k + 1u) {
    i2 = ((wg_uc[((k) & 3u)]) & (63u));
    i3 = 0u;
    if ((wid.x < cfg.n_queries)) {
    i4 = 0u;
    for (k2 = 0u; k2 < 32u; k2 = k2 + 1u) {
    if (((queries[(((((wid.x) & (65535u)) * 2048u) + ((lid.x * 32u) + k2)))] - index_f32[(((((i2 * 2048u) + ((lid.x * 32u) + k2))) & (8388607u)))]) > 0.0f)) {
    i3 = ((i3) | (u32((1u << ((u32(k2)) & 31u)))));
    }
    }
    }
    wg_ud[((((k * 64u) + lid.x)) & 255u)] = i3;
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i5 = ((wg_uc[((k) & 3u)]) & (63u));
    i6 = min(atomicLoad(&index_u32[(i5)]), 1792u);
    i7 = 0u;
    for (k2 = 0u; k2 < 28u; k2 = k2 + 1u) {
    if ((((k2 * 64u) + lid.x) < i6)) {
    i9 = 0u;
    i10 = 0u;
    for (k3 = 0u; k3 < 64u; k3 = k3 + 1u) {
    i9 = (i9 + countOneBits((u32(atomicLoad(&index_u32[(((((114752u + (((i5 * 1792u) + ((k2 * 64u) + lid.x)) * 64u)) + k3)) & (8388607u)))])) ^ u32(wg_ud[((((k * 64u) + k3)) & 255u)]))));
    }
    i8 = min(i9, 2047u);
    _ = atomicAdd(&wg_au_a[((i8) & 2047u)], 1u);
    }
    }
    }
    }
    workgroupBarrier();
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(2048u)), 16777216u); k = k + 1u) {
    if ((!b0)) {
    i11 = atomicLoad(&wg_au_a[((k) & 2047u)]);
    if (((i12 + i11) >= 64u)) {
    i0 = k;
    i13 = i12;
    b0 = true;
    } else {
    i12 = (i12 + i11);
    }
    }
    }
    if ((!b0)) {
    i13 = i12;
    }
    wg_ue[0u] = i0;
    wg_uf[0u] = i13;
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i14 = ((wg_uc[((k) & 3u)]) & (63u));
    i15 = min(atomicLoad(&index_u32[(i14)]), 1792u);
    i16 = 0u;
    for (k2 = 0u; k2 < 28u; k2 = k2 + 1u) {
    if ((((k2 * 64u) + lid.x) < i15)) {
    i9 = 0u;
    i10 = 0u;
    for (k3 = 0u; k3 < 64u; k3 = k3 + 1u) {
    i9 = (i9 + countOneBits((u32(atomicLoad(&index_u32[(((((114752u + (((i14 * 1792u) + ((k2 * 64u) + lid.x)) * 64u)) + k3)) & (8388607u)))])) ^ u32(wg_ud[((((k * 64u) + k3)) & 255u)]))));
    }
    i17 = min(i9, 2047u);
    i18 = 64u;
    if ((i17 < wg_ue[0u])) {
    i18 = atomicAdd(&wg_au_b[0u], 1u);
    } else {
    if ((i17 == wg_ue[0u])) {
    i18 = (wg_uf[0u] + atomicAdd(&wg_au_c[0u], 1u));
    }
    }
    if ((i18 < 64u)) {
    cands[(((((wid.x * 64u) + i18)) & (8388607u)))] = (atomicLoad(&index_u32[(((((64u + (i14 * 1792u)) + ((k2 * 64u) + lid.x))) & (8388607u)))]));
    }
    }
    }
    }
    }
}
