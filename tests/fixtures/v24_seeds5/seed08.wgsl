// grid 256 1 1 1
// entry ingest grid 256 1 1 1
// entry search grid 256 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_au_a: array<atomic<u32>, 257>;
var<workgroup> wg_au_b: array<atomic<u32>, 1>;
var<workgroup> wg_ua: array<u32, 256>;
var<workgroup> wg_ub: array<u32, 8>;
var<workgroup> wg_uc: array<u32, 1>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    for (k = 0u; k < 64u; k = k + 1u) {
    i1 = 0u;
    i1 = ((u32((u32(((lid.x * 64u) + k)) ^ ((lit_u32(291u) << ((lit_u32(16u)) & 31u)) | lit_u32(17767u)))) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i0 = i1;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (1u)) == 1u)), select(-1.0f, 1.0f, (((((i0) >> ((((k2 * 4u) + 1u)) & 31u))) & (1u)) == 1u))), select(-1.0f, 1.0f, (((((i0) >> ((((k2 * 4u) + 2u)) & 31u))) & (1u)) == 1u))), select(-1.0f, 1.0f, (((((i0) >> ((((k2 * 4u) + 3u)) & 31u))) & (1u)) == 1u)));
    f0 = (f0 + dot(keys[((((((((wid.x * 512u)) & (4194303u)) + (k * 8u)) + k2)) & (4194303u)))], v0));
    }
    }
    }
    wg_ua[((lid.x) & 255u)] = select(0u, 1u, (f0 > 0.0f));
    workgroupBarrier();
    if (((lid.x < 8u) & ((cfg.arrived_begin + wid.x) < cfg.arrived_end))) {
    for (k = 0u; k < 32u; k = k + 1u) {
    i2 = ((i2) | (u32((u32(wg_ua[((((lid.x * 32u) + k)) & 255u)]) << ((u32(k)) & 31u)))));
    }
    index_u32[((((((((cfg.arrived_begin + wid.x)) & (32767u)) * 8u) + lid.x)) & (4194303u)))] = (i2);
    }
}
@compute @workgroup_size(256, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0);
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
    var b0: bool = false;
    var b1: bool = false;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    i0 = 256u;
    for (k = 0u; k < 64u; k = k + 1u) {
    i2 = 0u;
    i2 = ((u32((u32(((lid.x * 64u) + k)) ^ ((lit_u32(291u) << ((lit_u32(16u)) & 31u)) | lit_u32(17767u)))) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((13u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((16u) & 31u))));
    i1 = i2;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(select(-1.0f, 1.0f, (((((i1) >> (((k2 * 4u)) & 31u))) & (1u)) == 1u)), select(-1.0f, 1.0f, (((((i1) >> ((((k2 * 4u) + 1u)) & 31u))) & (1u)) == 1u))), select(-1.0f, 1.0f, (((((i1) >> ((((k2 * 4u) + 2u)) & 31u))) & (1u)) == 1u))), select(-1.0f, 1.0f, (((((i1) >> ((((k2 * 4u) + 3u)) & 31u))) & (1u)) == 1u)));
    f0 = (f0 + dot(queries[((((((((wid.x) & (4095u)) * 512u) + (k * 8u)) + k2)) & (2097151u)))], v0));
    }
    }
    wg_ua[((lid.x) & 255u)] = select(0u, 1u, (f0 > 0.0f));
    atomicStore(&wg_au_a[min((lid.x), 256u)], 0u);
    if ((lid.x == 0u)) {
    atomicStore(&wg_au_a[256u], 0u);
    atomicStore(&wg_au_b[0u], 0u);
    }
    workgroupBarrier();
    if ((lid.x < 8u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    i3 = ((i3) | (u32((u32(wg_ua[((((lid.x * 32u) + k)) & 255u)]) << ((u32(k)) & 31u)))));
    }
    wg_ub[((lid.x) & 7u)] = i3;
    }
    workgroupBarrier();
    for (k = 0u; k < min((lit_u32(25600u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    i4 = 0u;
    i5 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i4 = (i4 + countOneBits((u32(index_u32[((((((((k + lid.x)) & (32767u)) * 8u) + k2)) & (4194303u)))]) ^ u32(wg_ub[((k2) & 7u)]))));
    }
    _ = atomicAdd(&wg_au_a[min((i4), 256u)], 1u);
    }
    }
    workgroupBarrier();
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(257u)), 16777216u); k = k + 1u) {
    b0 = false;
    i6 = (i6 + atomicLoad(&wg_au_a[min((k), 256u)]));
    if ((i6 >= 64u)) {
    i0 = k;
    b0 = true;
    }
    if (b0) { break; }
    }
    wg_uc[0u] = i0;
    }
    workgroupBarrier();
    i7 = wg_uc[0u];
    for (k = 0u; k < 2u; k = k + 1u) {
    i8 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(25600u)), 16777216u); k2 = k2 + 256u) {
    if (((k2 + lid.x) < cfg.arrived_end)) {
    i9 = 0u;
    i10 = 0u;
    for (k3 = 0u; k3 < 8u; k3 = k3 + 1u) {
    i9 = (i9 + countOneBits((u32(index_u32[((((((((k2 + lid.x)) & (32767u)) * 8u) + k3)) & (4194303u)))]) ^ u32(wg_ub[((k3) & 7u)]))));
    }
    b1 = (((k == 0u) & (i9 < i7)) | ((k == 1u) & (i9 == i7)));
    if (b1) {
    i11 = atomicAdd(&wg_au_b[0u], 1u);
    if (((i11 < 64u) & (wid.x < cfg.n_queries))) {
    cands[(((wid.x * 64u) + i11))] = ((k2 + lid.x));
    }
    }
    }
    }
    workgroupBarrier();
    }
    return;
}
