// grid 1 1 1 1
// entry ingest grid 1 1 1 1
// entry search grid 16 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_ua: array<u32, 1024>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(1, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    return;
}
@compute @workgroup_size(16, 1, 1)
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
    var v0: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var i6: u32 = 0u;
    var i7: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pua: array<u32, 64>;
    for (t = 0u; t < min((lit_u32(64u)), 16777216u); t = t + 1u) {
    pa[((t) & 63u)] = -3e38f;
    pua[((t) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    f0 = (f0 + dot(queries[(((wid.x * 512u) + k))], queries[(((wid.x * 512u) + k))]));
    }
    for (k = 0u; k < min((cfg.arrived_end), 16777216u); k = k + 16u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    f1 = 0.0f;
    f2 = 0.0f;
    i0 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    v0 = keys[((((((k + lid.x)) & (524287u)) * 512u) + k2))];
    f1 = (f1 + dot(v0, queries[(((wid.x * 512u) + k2))]));
    f2 = (f2 + dot(v0, v0));
    }
    f3 = -3e38f;
    if ((sqrt((f0 * f2)) > 0.0f)) {
    f3 = (f1 / sqrt((f0 * f2)));
    }
    if ((f3 > pa[63u])) {
    f4 = f3;
    i1 = (k + lid.x);
    i2 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    if ((f4 > pa[((k2) & 63u)])) {
    f5 = pa[((k2) & 63u)];
    i3 = pua[((k2) & 63u)];
    pa[((k2) & 63u)] = f4;
    pua[((k2) & 63u)] = i1;
    f4 = f5;
    i1 = i3;
    }
    }
    }
    }
    }
    }
    for (t = 0u; t < min((lit_u32(64u)), 16777216u); t = t + 1u) {
    wg_a[((((lid.x * 64u) + t)) & 1023u)] = pa[((t) & 63u)];
    wg_ua[((((lid.x * 64u) + t)) & 1023u)] = pua[((t) & 63u)];
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    pa[((k) & 63u)] = -3e38f;
    pua[((k) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < min((lit_u32(1024u)), 16777216u); k = k + 1u) {
    f6 = wg_a[((k) & 1023u)];
    i4 = wg_ua[((k) & 1023u)];
    if (((i4 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) && (f6 > pa[63u]))) {
    f7 = f6;
    i5 = i4;
    i6 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    if ((f7 > pa[((k2) & 63u)])) {
    f8 = pa[((k2) & 63u)];
    i7 = pua[((k2) & 63u)];
    pa[((k2) & 63u)] = f7;
    pua[((k2) & 63u)] = i5;
    f7 = f8;
    i5 = i7;
    }
    }
    }
    }
    cands[(((wid.x * 64u) + lid.x))] = (pua[((lid.x) & 63u)]);
    cands[((((wid.x * 64u) + 16u) + lid.x))] = (pua[(((16u + lid.x)) & 63u)]);
    cands[((((wid.x * 64u) + 32u) + lid.x))] = (pua[(((32u + lid.x)) & 63u)]);
    cands[((((wid.x * 64u) + 48u) + lid.x))] = (pua[(((48u + lid.x)) & 63u)]);
    }
}
