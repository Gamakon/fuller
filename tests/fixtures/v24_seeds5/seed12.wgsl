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
var<workgroup> wg_au_a: array<atomic<u32>, 264>;
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
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 1u) {
    f0 = keys[(((((wid.x) & (8191u)) * 2048u) + t))];
    i1 = 0u;
    i1 = ((((t * 8u) + ((lid.x) >> ((5u) & 31u))) + 1u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ ((lit_u32(34283u) << ((lit_u32(16u)) & 31u)) | lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(10618u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(11577u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i0 = i1;
    if ((((((i0) >> ((lid.x) & 31u))) & (1u)) == 1u)) {
    f1 = (f1 + f0);
    } else {
    f1 = (f1 - f0);
    }
    }
    if ((f1 > 0.0f)) {
    wg_ua[((lid.x) & 255u)] = 1u;
    } else {
    wg_ua[((lid.x) & 255u)] = 0u;
    }
    workgroupBarrier();
    if ((lid.x < 8u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    i2 = ((i2) | (u32((u32(wg_ua[((((lid.x * 32u) + k)) & 255u)]) << ((u32(k)) & 31u)))));
    }
    index_u32[((((((cfg.arrived_begin + wid.x)) & (32767u)) * 8u) + lid.x))] = (i2);
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
    var b0: bool = false;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pi: array<u32, 16>;
    i0 = 256u;
    for (t = 0u; t < min((lit_u32(2048u)), 16777216u); t = t + 1u) {
    f0 = queries[(((((((wid.x) & (65535u)) * 2048u) + t)) & (((u32((lit_u32(2047u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))))];
    i2 = 0u;
    i2 = ((((t * 8u) + ((lid.x) >> ((5u) & 31u))) + 1u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i2 = u32((u32(i2) ^ ((lit_u32(34283u) << ((lit_u32(16u)) & 31u)) | lit_u32(51819u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((12u) & 31u))));
    i2 = (i2 * ((u32((lit_u32(10618u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(11577u))));
    i2 = u32((u32(i2) ^ (u32(i2) >> ((15u) & 31u))));
    i1 = i2;
    if ((((((i1) >> ((lid.x) & 31u))) & (1u)) == 1u)) {
    f1 = (f1 + f0);
    } else {
    f1 = (f1 - f0);
    }
    }
    if ((f1 > 0.0f)) {
    wg_ua[((lid.x) & 255u)] = 1u;
    } else {
    wg_ua[((lid.x) & 255u)] = 0u;
    }
    i3 = lid.x;
    for (t = 0u; t < min((lit_u32(264u)), 16777216u); t = t + 256u) {
    if (((lid.x + t) < 264u)) {
    atomicStore(&wg_au_a[min(((lid.x + t)), 263u)], 0u);
    }
    }
    if ((lid.x == 0u)) {
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
    pi[0u] = wg_ub[0u];
    pi[1u] = wg_ub[1u];
    pi[2u] = wg_ub[2u];
    pi[3u] = wg_ub[3u];
    pi[4u] = wg_ub[4u];
    pi[5u] = wg_ub[5u];
    pi[6u] = wg_ub[6u];
    pi[7u] = wg_ub[7u];
    for (k = 0u; k < min((lit_u32(32768u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < min(cfg.arrived_end, 32768u))) {
    i5 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i5 = (i5 + countOneBits((u32(index_u32[((((k + lid.x) * 8u) + k2))]) ^ u32(pi[((k2) & 15u)]))));
    }
    _ = atomicAdd(&wg_au_a[min((((i5) & (255u))), 263u)], 1u);
    if ((i5 > 255u)) {
    _ = atomicAdd(&wg_au_a[256u], 1u);
    }
    }
    }
    workgroupBarrier();
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(257u)), 16777216u); k = k + 1u) {
    if ((!b0)) {
    i6 = (i6 + atomicLoad(&wg_au_a[min((k), 263u)]));
    if ((i6 >= 64u)) {
    i0 = k;
    b0 = true;
    }
    }
    }
    wg_uc[0u] = i0;
    }
    workgroupBarrier();
    i7 = wg_uc[0u];
    for (k = 0u; k < min((lit_u32(32768u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < min(cfg.arrived_end, 32768u))) {
    i8 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i8 = (i8 + countOneBits((u32(index_u32[((((k + lid.x) * 8u) + k2))]) ^ u32(pi[((k2) & 15u)]))));
    }
    if ((i8 < i7)) {
    i9 = atomicAdd(&wg_au_b[0u], 1u);
    if ((i9 < 64u)) {
    cands[(((((wid.x) & (65535u)) * 64u) + i9))] = ((k + lid.x));
    }
    }
    }
    }
    workgroupBarrier();
    for (k = 0u; k < min((lit_u32(32768u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < min(cfg.arrived_end, 32768u))) {
    i10 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    i10 = (i10 + countOneBits((u32(index_u32[((((k + lid.x) * 8u) + k2))]) ^ u32(pi[((k2) & 15u)]))));
    }
    if ((i10 == i7)) {
    i11 = atomicAdd(&wg_au_b[0u], 1u);
    if ((i11 < 64u)) {
    cands[(((((wid.x) & (65535u)) * 64u) + i11))] = ((k + lid.x));
    }
    }
    }
    }
}
