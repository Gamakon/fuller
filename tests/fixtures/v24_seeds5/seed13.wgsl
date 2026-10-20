// grid 64 1 1 1
// entry ingest grid 64 1 1 1
// entry search grid 128 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_b: array<f32, 1024>;
var<workgroup> wg_ua: array<u32, 1024>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(64, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    if ((cfg.pass_index == 0u)) {
    for (k = 0u; k < 64u; k = k + 1u) {
    i1 = 0u;
    i1 = u32(((u32((lid.x * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))))) ^ u32((k * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u)))))) ^ ((lit_u32(291u) << ((lit_u32(16u)) & 31u)) | lit_u32(17767u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(10618u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(11577u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i0 = i1;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    f0 = (f0 + dot(keys[((((((wid.x) & (8191u)) * 512u) + (k * 8u)) + k2))], vec4<f32>(vec3<f32>(vec2<f32>(select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (1u)) != 0u)), select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (2u)) != 0u))), select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (4u)) != 0u))), select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (8u)) != 0u)))));
    }
    }
    index_f32[((((((cfg.arrived_begin + wid.x)) & (32767u)) * 64u) + lid.x))] = (f0);
    }
    wg_a[((lid.x) & 63u)] = f0;
    workgroupBarrier();
    if (((lid.x == 0u) & (cfg.pass_index == 0u))) {
    for (k = 0u; k < 64u; k = k + 1u) {
    f1 = (f1 + (wg_a[((k) & 63u)] * wg_a[((k) & 63u)]));
    }
    if ((f1 > 0.0f)) {
    f2 = inverseSqrt(f1);
    }
    index_f32[((2097152u + (((cfg.arrived_begin + wid.x)) & (32767u))))] = (f2);
    }
}
@compute @workgroup_size(128, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var i7: u32 = 0u;
    var i8: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    if ((lid.x < 64u)) {
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 64u; k = k + 1u) {
    i1 = 0u;
    i1 = u32(((u32((lid.x * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))))) ^ u32((k * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u)))))) ^ ((lit_u32(291u) << ((lit_u32(16u)) & 31u)) | lit_u32(17767u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(10618u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(11577u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i0 = i1;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    f0 = (f0 + dot(queries[((((((wid.x) & (4095u)) * 512u) + (k * 8u)) + k2))], vec4<f32>(vec3<f32>(vec2<f32>(select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (1u)) != 0u)), select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (2u)) != 0u))), select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (4u)) != 0u))), select(-1.0f, 1.0f, (((((i0) >> (((k2 * 4u)) & 31u))) & (8u)) != 0u)))));
    }
    }
    }
    wg_a[((lid.x) & 63u)] = f0;
    }
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    pa[((k) & 63u)] = -3e38f;
    pi[((k) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(32768u)), 16777216u); k = k + 128u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    f1 = 0.0f;
    i2 = 0u;
    for (k2 = 0u; k2 < 64u; k2 = k2 + 1u) {
    f1 = (f1 + (index_f32[((((((k + lid.x)) & (32767u)) * 64u) + k2))] * wg_a[((k2) & 63u)]));
    }
    f2 = (f1 * index_f32[((2097152u + (((k + lid.x)) & (32767u))))]);
    if ((f2 > pa[7u])) {
    f3 = f2;
    i3 = (k + lid.x);
    i4 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    if ((f3 > pa[((k2) & 63u)])) {
    f4 = pa[((k2) & 63u)];
    i5 = pi[((k2) & 15u)];
    pa[((k2) & 63u)] = f3;
    pi[((k2) & 15u)] = i3;
    f3 = f4;
    i3 = i5;
    }
    }
    }
    }
    }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    wg_b[((((lid.x * 8u) + k)) & 1023u)] = pa[((k) & 63u)];
    wg_ua[((((lid.x * 8u) + k)) & 1023u)] = pi[((k) & 15u)];
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 8u; k = k + 1u) {
    i6 = wg_ua[(((lid.x + (k * 128u))) & 1023u)];
    if ((i6 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    f5 = wg_b[(((lid.x + (k * 128u))) & 1023u)];
    i7 = 0u;
    i8 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(1024u)), 16777216u); k2 = k2 + 1u) {
    if (((wg_ua[((k2) & 1023u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & ((wg_b[((k2) & 1023u)] > f5) | ((wg_b[((k2) & 1023u)] == f5) & (k2 < (lid.x + (k * 128u))))))) {
    i7 = (i7 + 1u);
    }
    }
    if ((i7 < 64u)) {
    cands[(((wid.x * 64u) + i7))] = (i6);
    }
    }
    }
    }
}
