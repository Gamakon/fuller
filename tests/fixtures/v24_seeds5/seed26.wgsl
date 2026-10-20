// grid 1 1 1 1
// entry ingest grid 1 1 1 1
// entry search grid 32 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(1, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    f0 = 0.0f;
    i1 = 0u;
    i2 = 0u;
    i3 = 0u;
    for (k = 0u; k < 6u; k = k + 1u) {
    f0 = 0.0f;
    i1 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    f1 = keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + (((((k * 317u) + (k2 * 241u)) + 19u)) & (2047u))))];
    if (((((k + k2)) & (1u)) == 0u)) {
    f0 = (f0 + f1);
    } else {
    f0 = (f0 - f1);
    }
    }
    if ((f0 > 0.0f)) {
    i2 = ((i2) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    i0 = i2;
    i4 = atomicAdd(&index_u32[((0u + i0))], (1u));
    if ((i4 < 512u)) {
    atomicStore(&index_u32[(((64u + (i0 * 512u)) + i4))], ((cfg.arrived_begin + wid.x)));
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    index_f32[((((cfg.arrived_begin + wid.x) * 64u) + k))] = (keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + ((((k * 31u) + 17u)) & (2047u))))]);
    }
    }
    }
}
@compute @workgroup_size(32, 1, 1)
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
    var i7: u32 = 0u;
    var i8: u32 = 0u;
    var i9: u32 = 0u;
    var i10: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    f0 = -3e38f;
    f1 = -3e38f;
    i0 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    i1 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if ((wid.x < cfg.n_queries)) {
    cands[(((wid.x * 64u) + lid.x))] = (((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))));
    cands[((((wid.x * 64u) + 32u) + lid.x))] = (((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))));
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    f2 = (f2 + (queries[(((wid.x * 2048u) + ((((k * 31u) + 17u)) & (2047u))))] * queries[(((wid.x * 2048u) + ((((k * 31u) + 17u)) & (2047u))))]));
    }
    if ((f2 > 0.0f)) {
    f3 = 0.0f;
    i3 = 0u;
    i4 = 0u;
    i5 = 0u;
    for (k = 0u; k < 6u; k = k + 1u) {
    f3 = 0.0f;
    i3 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    f4 = queries[(((wid.x * 2048u) + (((((k * 317u) + (k2 * 241u)) + 19u)) & (2047u))))];
    if (((((k + k2)) & (1u)) == 0u)) {
    f3 = (f3 + f4);
    } else {
    f3 = (f3 - f4);
    }
    }
    if ((f3 > 0.0f)) {
    i4 = ((i4) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    i2 = i4;
    for (k = 0u; k < 4u; k = k + 1u) {
    i6 = i2;
    if ((k == 1u)) {
    i6 = u32((u32(i2) ^ 1u));
    }
    if ((k == 2u)) {
    i6 = u32((u32(i2) ^ 2u));
    }
    if ((k == 3u)) {
    i6 = u32((u32(i2) ^ 4u));
    }
    i7 = min(atomicLoad(&index_u32[((0u + i6))]), 512u);
    i8 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 32u) {
    if (((k2 + lid.x) < i7)) {
    i9 = atomicLoad(&index_u32[(((64u + (i6 * 512u)) + (k2 + lid.x)))]);
    if ((i9 < cfg.arrived_end)) {
    f5 = 0.0f;
    f6 = 0.0f;
    i10 = 0u;
    for (k3 = 0u; k3 < min((lit_u32(64u)), 16777216u); k3 = k3 + 1u) {
    f7 = index_f32[(((i9 * 64u) + k3))];
    f5 = (f5 + (f7 * queries[(((wid.x * 2048u) + ((((k3 * 31u) + 17u)) & (2047u))))]));
    f6 = (f6 + (f7 * f7));
    }
    if ((sqrt((f2 * f6)) > 0.0f)) {
    if (((f5 / sqrt((f2 * f6))) > f0)) {
    f1 = f0;
    i1 = i0;
    f0 = (f5 / sqrt((f2 * f6)));
    i0 = i9;
    } else {
    if (((f5 / sqrt((f2 * f6))) > f1)) {
    f1 = (f5 / sqrt((f2 * f6)));
    i1 = i9;
    }
    }
    }
    }
    }
    }
    }
    if ((i0 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    cands[(((wid.x * 64u) + lid.x))] = (i0);
    }
    if ((i1 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    cands[((((wid.x * 64u) + 32u) + lid.x))] = (i1);
    }
    }
    }
}
