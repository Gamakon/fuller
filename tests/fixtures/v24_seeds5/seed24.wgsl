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
var<workgroup> wg_b: array<f32, 64>;
var<workgroup> wg_c: array<f32, 1>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var k: u32 = 0u;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    f0 = (f0 + keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + ((lid.x * 32u) + k)))]);
    }
    wg_a[((lid.x) & 63u)] = f0;
    index_f32[((((cfg.arrived_begin + wid.x) * 64u) + lid.x))] = (f0);
    }
    workgroupBarrier();
    if (((lid.x == 0u) & ((cfg.arrived_begin + wid.x) < cfg.arrived_end))) {
    f1 = abs(wg_a[0u]);
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    if ((wg_a[((k) & 63u)] >= 0.0f)) {
    i0 = ((i0) | (u32((1u << ((u32(k)) & 31u)))));
    }
    f2 = abs(wg_a[((k) & 63u)]);
    if ((f2 < f1)) {
    f1 = f2;
    i1 = k;
    }
    }
    i2 = atomicAdd(&index_u32[((0u + i0))], (1u));
    if ((i2 < 512u)) {
    atomicStore(&index_u32[(((1024u + (i0 * 512u)) + i2))], ((cfg.arrived_begin + wid.x)));
    }
    i3 = atomicAdd(&index_u32[((0u + u32((u32(i0) ^ (1u << ((u32(i1)) & 31u))))))], (1u));
    if ((i3 < 512u)) {
    atomicStore(&index_u32[(((1024u + (u32((u32(i0) ^ (1u << ((u32(i1)) & 31u)))) * 512u)) + i3))], ((cfg.arrived_begin + wid.x)));
    }
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
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var b0: bool = false;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    f0 = -3e38f;
    i0 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    f1 = (f1 + queries[(((wid.x * 2048u) + ((lid.x * 32u) + k)))]);
    }
    wg_b[((lid.x) & 63u)] = f1;
    }
    workgroupBarrier();
    if (((lid.x == 0u) & (wid.x < cfg.n_queries))) {
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    f2 = (f2 + (wg_b[((k) & 63u)] * wg_b[((k) & 63u)]));
    }
    wg_c[0u] = f2;
    }
    workgroupBarrier();
    if (((wid.x < cfg.n_queries) & (wg_c[0u] > 0.0f))) {
    f3 = abs(wg_b[0u]);
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    if ((wg_b[((k) & 63u)] >= 0.0f)) {
    i1 = ((i1) | (u32((1u << ((u32(k)) & 31u)))));
    }
    f4 = abs(wg_b[((k) & 63u)]);
    if ((f4 < f3)) {
    f3 = f4;
    i2 = k;
    }
    }
    i3 = min(atomicLoad(&index_u32[((0u + i1))]), 512u);
    i4 = min(atomicLoad(&index_u32[((0u + u32((u32(i1) ^ (1u << ((u32(i2)) & 31u))))))]), 512u);
    for (k = 0u; k < min((lit_u32(1024u)), 16777216u); k = k + 64u) {
    b0 = false;
    i5 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if (((k + lid.x) < 512u)) {
    if (((k + lid.x) < i3)) {
    i5 = atomicLoad(&index_u32[(((1024u + (i1 * 512u)) + (k + lid.x)))]);
    b0 = true;
    }
    } else {
    if ((select(((k + lid.x)) - (512u), 0u, (512u) > ((k + lid.x))) < i4)) {
    i5 = atomicLoad(&index_u32[(((1024u + (u32((u32(i1) ^ (1u << ((u32(i2)) & 31u)))) * 512u)) + select(((k + lid.x)) - (512u), 0u, (512u) > ((k + lid.x)))))]);
    b0 = true;
    }
    }
    if ((b0 & (i5 < cfg.arrived_end))) {
    f5 = 0.0f;
    f6 = 0.0f;
    i6 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    f7 = index_f32[(((i5 * 64u) + k2))];
    f5 = (f5 + (f7 * wg_b[((k2) & 63u)]));
    f6 = (f6 + (f7 * f7));
    }
    if ((f6 > 0.0f)) {
    if (((f5 / sqrt((f6 * wg_c[0u]))) > f0)) {
    f0 = (f5 / sqrt((f6 * wg_c[0u])));
    i0 = i5;
    }
    }
    }
    }
    if ((i0 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    cands[(((wid.x * 64u) + lid.x))] = (i0);
    }
    }
}
