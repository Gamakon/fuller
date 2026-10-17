// grid 1 1 1 1
// entry ingest grid 1 1 1 1
// entry search grid 8 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_ua: array<u32, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(1, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var f6: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var k: u32 = 0u;
    if ((((cfg.arrived_begin + wid.x) < cfg.arrived_end) && (cfg.pass_index == 0u))) {
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    i1 = 0u;
    i2 = 0u;
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    f6 = dot(keys[((((((cfg.arrived_begin + wid.x)) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, -1.0f), 1.0f), -1.0f));
    f0 = (f0 + f6);
    f1 = (f1 + (f6 * select(-1.0f, 1.0f, (((k) & (1u)) == 0u))));
    f2 = (f2 + (f6 * select(-1.0f, 1.0f, (((k) & (2u)) == 0u))));
    f3 = (f3 + (f6 * select(-1.0f, 1.0f, (((k) & (4u)) == 0u))));
    f4 = (f4 + (f6 * select(-1.0f, 1.0f, (((k) & (8u)) == 0u))));
    f5 = (f5 + (f6 * select(-1.0f, 1.0f, (((k) & (16u)) == 0u))));
    }
    if ((f0 >= 0.0f)) {
    i2 = ((i2) | (1u));
    }
    if ((f1 >= 0.0f)) {
    i2 = ((i2) | (2u));
    }
    if ((f2 >= 0.0f)) {
    i2 = ((i2) | (4u));
    }
    if ((f3 >= 0.0f)) {
    i2 = ((i2) | (8u));
    }
    if ((f4 >= 0.0f)) {
    i2 = ((i2) | (16u));
    }
    if ((f5 >= 0.0f)) {
    i2 = ((i2) | (32u));
    }
    i0 = i2;
    index_u32[((((cfg.arrived_begin + wid.x)) & (8388607u)))] = (i0);
    }
}
@compute @workgroup_size(8, 1, 1)
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
    var f9: f32 = 0.0;
    var f10: f32 = 0.0;
    var f11: f32 = 0.0;
    var f12: f32 = 0.0;
    var f13: f32 = 0.0;
    var f14: f32 = 0.0;
    var f15: f32 = 0.0;
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
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pc: array<f32, 64>;
    var pua: array<u32, 64>;
    for (t = 0u; t < min((lit_u32(64u)), 16777216u); t = t + 1u) {
    pa[((t) & 63u)] = -3e38f;
    pua[((t) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    if ((wid.x < cfg.n_queries)) {
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    i1 = 0u;
    i2 = 0u;
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    f6 = dot(queries[(((wid.x * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, -1.0f), 1.0f), -1.0f));
    f0 = (f0 + f6);
    f1 = (f1 + (f6 * select(-1.0f, 1.0f, (((k) & (1u)) == 0u))));
    f2 = (f2 + (f6 * select(-1.0f, 1.0f, (((k) & (2u)) == 0u))));
    f3 = (f3 + (f6 * select(-1.0f, 1.0f, (((k) & (4u)) == 0u))));
    f4 = (f4 + (f6 * select(-1.0f, 1.0f, (((k) & (8u)) == 0u))));
    f5 = (f5 + (f6 * select(-1.0f, 1.0f, (((k) & (16u)) == 0u))));
    }
    if ((f0 >= 0.0f)) {
    i2 = ((i2) | (1u));
    }
    if ((f1 >= 0.0f)) {
    i2 = ((i2) | (2u));
    }
    if ((f2 >= 0.0f)) {
    i2 = ((i2) | (4u));
    }
    if ((f3 >= 0.0f)) {
    i2 = ((i2) | (8u));
    }
    if ((f4 >= 0.0f)) {
    i2 = ((i2) | (16u));
    }
    if ((f5 >= 0.0f)) {
    i2 = ((i2) | (32u));
    }
    i0 = i2;
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    f7 = (f7 + dot(queries[(((wid.x * 512u) + k))], queries[(((wid.x * 512u) + k))]));
    }
    for (k = 0u; k < min((cfg.arrived_end), 16777216u); k = k + 8u) {
    if (((k + lid.x) < cfg.arrived_end)) {
    if (((((index_u32[((((k + lid.x)) & (8388607u)))] == i0) || (index_u32[((((k + lid.x)) & (8388607u)))] == u32((u32(i0) ^ 1u)))) || (index_u32[((((k + lid.x)) & (8388607u)))] == u32((u32(i0) ^ 2u)))) || (index_u32[((((k + lid.x)) & (8388607u)))] == u32((u32(i0) ^ 4u))))) {
    f8 = 0.0f;
    f9 = 0.0f;
    i3 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    v0 = keys[((((((k + lid.x)) & (524287u)) * 512u) + k2))];
    f8 = (f8 + dot(v0, queries[(((wid.x * 512u) + k2))]));
    f9 = (f9 + dot(v0, v0));
    }
    f10 = -3e38f;
    if ((sqrt((f7 * f9)) > 0.0f)) {
    f10 = (f8 / sqrt((f7 * f9)));
    }
    if ((f10 > pa[63u])) {
    f11 = f10;
    i4 = (k + lid.x);
    i5 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    if ((f11 > pc[((k2) & 63u)])) {
    f12 = pa[((k2) & 63u)];
    i6 = pua[((k2) & 63u)];
    pa[((k2) & 63u)] = f11;
    pua[((k2) & 63u)] = i4;
    f11 = f12;
    i4 = i6;
    }
    }
    }
    }
    }
    }
    }
    for (t = 0u; t < min((lit_u32(64u)), 16777216u); t = t + 1u) {
    wg_a[((((lid.x * 64u) + t)) & 511u)] = pa[((t) & 63u)];
    wg_ua[((((lid.x * 64u) + t)) & 511u)] = pua[((t) & 63u)];
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    pa[((k) & 63u)] = -3e38f;
    pua[((k) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    f13 = wg_a[((k) & 511u)];
    i7 = wg_ua[((k) & 511u)];
    if (((i7 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) && (f13 > pa[63u]))) {
    f14 = f13;
    i8 = i7;
    i9 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    if ((f14 > pa[((k2) & 63u)])) {
    f15 = pa[((k2) & 63u)];
    i10 = pua[((k2) & 63u)];
    pa[((k2) & 63u)] = f14;
    pua[((k2) & 63u)] = i8;
    f14 = f15;
    i8 = i10;
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 8u) {
    cands[((((wid.x * 64u) + k) + lid.x))] = (pua[(((k + lid.x)) & 63u)]);
    }
    }
}
