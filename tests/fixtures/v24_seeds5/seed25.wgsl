// grid 1 1 1 1
// entry ingest grid 1 1 1 1
// entry search grid 1 1 1 1
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
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    i1 = 0u;
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 17u))] > 0.0f)) {
    i1 = ((i1) | (1u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 103u))] > 0.0f)) {
    i1 = ((i1) | (2u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 211u))] > 0.0f)) {
    i1 = ((i1) | (4u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 389u))] > 0.0f)) {
    i1 = ((i1) | (8u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 547u))] > 0.0f)) {
    i1 = ((i1) | (16u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 701u))] > 0.0f)) {
    i1 = ((i1) | (32u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 887u))] > 0.0f)) {
    i1 = ((i1) | (64u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 1049u))] > 0.0f)) {
    i1 = ((i1) | (128u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 1217u))] > 0.0f)) {
    i1 = ((i1) | (256u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 1399u))] > 0.0f)) {
    i1 = ((i1) | (512u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 1613u))] > 0.0f)) {
    i1 = ((i1) | (1024u));
    }
    if ((keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 2048u) + 1879u))] > 0.0f)) {
    i1 = ((i1) | (2048u));
    }
    i0 = i1;
    i2 = atomicAdd(&index_u32[((0u + ((i0) & (511u))))], (1u));
    if ((i2 < 32u)) {
    atomicStore(&index_u32[(((4608u + (((i0) & (511u)) * 32u)) + i2))], ((cfg.arrived_begin + wid.x)));
    }
    if ((i2 >= 12u)) {
    i3 = atomicAdd(&index_u32[((512u + ((i0) & (4095u))))], (1u));
    if ((i3 < 16u)) {
    atomicStore(&index_u32[(((20992u + (((i0) & (4095u)) * 16u)) + i3))], ((cfg.arrived_begin + wid.x)));
    }
    }
    }
}
@compute @workgroup_size(1, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
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
    var b0: bool = false;
    var b1: bool = false;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pua: array<u32, 64>;
    i1 = 0u;
    if ((queries[(((wid.x * 2048u) + 17u))] > 0.0f)) {
    i1 = ((i1) | (1u));
    }
    if ((queries[(((wid.x * 2048u) + 103u))] > 0.0f)) {
    i1 = ((i1) | (2u));
    }
    if ((queries[(((wid.x * 2048u) + 211u))] > 0.0f)) {
    i1 = ((i1) | (4u));
    }
    if ((queries[(((wid.x * 2048u) + 389u))] > 0.0f)) {
    i1 = ((i1) | (8u));
    }
    if ((queries[(((wid.x * 2048u) + 547u))] > 0.0f)) {
    i1 = ((i1) | (16u));
    }
    if ((queries[(((wid.x * 2048u) + 701u))] > 0.0f)) {
    i1 = ((i1) | (32u));
    }
    if ((queries[(((wid.x * 2048u) + 887u))] > 0.0f)) {
    i1 = ((i1) | (64u));
    }
    if ((queries[(((wid.x * 2048u) + 1049u))] > 0.0f)) {
    i1 = ((i1) | (128u));
    }
    if ((queries[(((wid.x * 2048u) + 1217u))] > 0.0f)) {
    i1 = ((i1) | (256u));
    }
    if ((queries[(((wid.x * 2048u) + 1399u))] > 0.0f)) {
    i1 = ((i1) | (512u));
    }
    if ((queries[(((wid.x * 2048u) + 1613u))] > 0.0f)) {
    i1 = ((i1) | (1024u));
    }
    if ((queries[(((wid.x * 2048u) + 1879u))] > 0.0f)) {
    i1 = ((i1) | (2048u));
    }
    i0 = i1;
    for (k = 0u; k < 4u; k = k + 1u) {
    i2 = i0;
    if ((k == 1u)) {
    i2 = u32((u32(i0) ^ 1u));
    }
    if ((k == 2u)) {
    i2 = u32((u32(i0) ^ 2u));
    }
    if ((k == 3u)) {
    i2 = u32((u32(i0) ^ 4u));
    }
    i3 = min(atomicLoad(&index_u32[((512u + i2))]), 16u);
    i4 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(16u)), 16777216u); k2 = k2 + 1u) {
    if (((k2 < i3) & (i5 < 64u))) {
    i6 = atomicLoad(&index_u32[(((20992u + (i2 * 16u)) + k2))]);
    b0 = false;
    i7 = 0u;
    for (k3 = 0u; k3 < 64u; k3 = k3 + 1u) {
    if (((k3 < i5) & (pua[((k3) & 63u)] == i6))) {
    b0 = true;
    }
    }
    if (((!b0) & (i6 < cfg.arrived_end))) {
    pua[((i5) & 63u)] = i6;
    i5 = (i5 + 1u);
    }
    }
    }
    }
    i8 = min(atomicLoad(&index_u32[((0u + ((i0) & (511u))))]), 32u);
    for (t = 0u; t < min((lit_u32(32u)), 16777216u); t = t + 1u) {
    if (((t < i8) & (i5 < 64u))) {
    i9 = atomicLoad(&index_u32[(((4608u + (((i0) & (511u)) * 32u)) + t))]);
    b1 = false;
    i10 = 0u;
    for (k = 0u; k < 64u; k = k + 1u) {
    if (((k < i5) & (pua[((k) & 63u)] == i9))) {
    b1 = true;
    }
    }
    if (((!b1) & (i9 < cfg.arrived_end))) {
    pua[((i5) & 63u)] = i9;
    i5 = (i5 + 1u);
    }
    }
    }
    for (k = 0u; k < 64u; k = k + 1u) {
    if ((k < i5)) {
    if ((!(wid.x >= cfg.n_queries))) {
    cands[(((wid.x * 64u) + k))] = (pua[((k) & 63u)]);
    }
    }
    }
}
