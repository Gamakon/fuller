// grid 1 1 1 1
// entry ingest grid 1 1 1 1
// entry search grid 8 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_b: array<f32, 64>;
var<workgroup> wg_ua: array<u32, 64>;
var<workgroup> wg_ub: array<u32, 64>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(1, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var k: u32 = 0u;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    i1 = 0u;
    i2 = 0u;
    for (k = 0u; k < 10u; k = k + 1u) {
    if ((select(select(select(dot(keys[((((((cfg.arrived_begin + wid.x)) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, 1.0f), 1.0f), -1.0f)), dot(keys[((((((cfg.arrived_begin + wid.x)) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, -1.0f), -1.0f), 1.0f)), (((k) & (3u)) == 2u)), dot(keys[((((((cfg.arrived_begin + wid.x)) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, 1.0f), -1.0f), -1.0f)), (((k) & (3u)) == 1u)), dot(keys[((((((cfg.arrived_begin + wid.x)) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, -1.0f), 1.0f), -1.0f)), (((k) & (3u)) == 0u)) > 0.0f)) {
    i1 = ((i1) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    i0 = ((i1) & (1023u));
    i3 = atomicAdd(&index_u32[((0u + i0))], (1u));
    if ((i3 < 32u)) {
    atomicStore(&index_u32[(((2048u + (i0 * 32u)) + i3))], ((cfg.arrived_begin + wid.x)));
    }
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
    var i11: u32 = 0u;
    var i12: u32 = 0u;
    var i13: u32 = 0u;
    var i14: u32 = 0u;
    var i15: u32 = 0u;
    var i16: u32 = 0u;
    var i17: u32 = 0u;
    var i18: u32 = 0u;
    var i19: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    var pua: array<u32, 64>;
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    pa[((t) & 63u)] = -3e38f;
    pi[((t) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    pb[((t) & 63u)] = -3e38f;
    pj[((t) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    if ((wid.x < cfg.n_queries)) {
    f1 = 0.0f;
    i0 = 0u;
    for (k = 0u; k < min((lit_u32(512u)), 16777216u); k = k + 1u) {
    f1 = (f1 + dot(queries[(((((wid.x) & (524287u)) * 512u) + k))], queries[(((((wid.x) & (524287u)) * 512u) + k))]));
    }
    f0 = f1;
    for (k = 0u; k < min((select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))), 16777216u); k = k + 8u) {
    if ((((cfg.arrived_begin + k) + lid.x) < cfg.arrived_end)) {
    f3 = 0.0f;
    f4 = 0.0f;
    i1 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    v0 = keys[(((((((cfg.arrived_begin + k) + lid.x)) & (524287u)) * 512u) + k2))];
    f3 = (f3 + dot(v0, queries[(((((wid.x) & (524287u)) * 512u) + k2))]));
    f4 = (f4 + dot(v0, v0));
    }
    if ((sqrt((f4 * f0)) > 0.0f)) {
    f2 = (f3 / sqrt((f4 * f0)));
    } else {
    f2 = -3e38f;
    }
    if ((f2 > pa[7u])) {
    f5 = f2;
    i2 = ((cfg.arrived_begin + k) + lid.x);
    i3 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(8u)), 16777216u); k2 = k2 + 1u) {
    if ((f5 > pa[((k2) & 63u)])) {
    f6 = pa[((k2) & 63u)];
    i4 = pi[((k2) & 15u)];
    pa[((k2) & 63u)] = f5;
    pi[((k2) & 15u)] = i2;
    f5 = f6;
    i2 = i4;
    }
    }
    }
    }
    }
    i6 = 0u;
    i7 = 0u;
    for (k = 0u; k < 10u; k = k + 1u) {
    if ((select(select(select(dot(queries[(((((wid.x) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, 1.0f), 1.0f), -1.0f)), dot(queries[(((((wid.x) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, -1.0f), -1.0f), 1.0f)), (((k) & (3u)) == 2u)), dot(queries[(((((wid.x) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, 1.0f), -1.0f), -1.0f)), (((k) & (3u)) == 1u)), dot(queries[(((((wid.x) & (524287u)) * 512u) + k))], vec4<f32>(vec3<f32>(vec2<f32>(1.0f, -1.0f), 1.0f), -1.0f)), (((k) & (3u)) == 0u)) > 0.0f)) {
    i6 = ((i6) | (u32((1u << ((u32(k)) & 31u)))));
    }
    }
    i5 = ((i6) & (1023u));
    i8 = min(atomicLoad(&index_u32[((0u + u32((u32(i5) ^ (1u << ((u32(lid.x)) & 31u))))))]), 32u);
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    if ((k < i8)) {
    i9 = atomicLoad(&index_u32[(((2048u + (u32((u32(i5) ^ (1u << ((u32(lid.x)) & 31u)))) * 32u)) + k))]);
    if ((i9 < cfg.arrived_begin)) {
    f3 = 0.0f;
    f4 = 0.0f;
    i1 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    v0 = keys[(((((i9) & (524287u)) * 512u) + k2))];
    f3 = (f3 + dot(v0, queries[(((((wid.x) & (524287u)) * 512u) + k2))]));
    f4 = (f4 + dot(v0, v0));
    }
    if ((sqrt((f4 * f0)) > 0.0f)) {
    f7 = (f3 / sqrt((f4 * f0)));
    } else {
    f7 = -3e38f;
    }
    if ((f7 > pb[7u])) {
    f8 = f7;
    i10 = i9;
    for (k2 = 0u; k2 < min((lit_u32(8u)), 16777216u); k2 = k2 + 1u) {
    if ((f8 > pb[((k2) & 63u)])) {
    f9 = pb[((k2) & 63u)];
    i11 = pj[((k2) & 15u)];
    pb[((k2) & 63u)] = f8;
    pj[((k2) & 15u)] = i10;
    f8 = f9;
    i10 = i11;
    }
    }
    }
    }
    }
    }
    }
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    wg_a[((((lid.x * 8u) + t)) & 63u)] = pa[((t) & 63u)];
    wg_ua[((((lid.x * 8u) + t)) & 63u)] = pi[((t) & 15u)];
    wg_b[((((lid.x * 8u) + t)) & 63u)] = pb[((t) & 63u)];
    wg_ub[((((lid.x * 8u) + t)) & 63u)] = pj[((t) & 15u)];
    }
    workgroupBarrier();
    if (((lid.x == 0u) && (wid.x < cfg.n_queries))) {
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    pc[((k) & 63u)] = -3e38f;
    pua[((k) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < 64u; k = k + 1u) {
    f10 = wg_a[((k) & 63u)];
    i12 = wg_ua[((k) & 63u)];
    if (((i12 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) && (f10 > pc[31u]))) {
    f11 = f10;
    i13 = i12;
    i14 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(32u)), 16777216u); k2 = k2 + 1u) {
    if ((f11 > pc[((k2) & 63u)])) {
    f12 = pc[((k2) & 63u)];
    i15 = pua[((k2) & 63u)];
    pc[((k2) & 63u)] = f11;
    pua[((k2) & 63u)] = i13;
    f11 = f12;
    i13 = i15;
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    cands[(((wid.x * 64u) + k))] = (pua[((k) & 63u)]);
    pc[((k) & 63u)] = -3e38f;
    pua[((k) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < 64u; k = k + 1u) {
    f13 = wg_b[((k) & 63u)];
    i16 = wg_ub[((k) & 63u)];
    if (((i16 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) && (f13 > pc[31u]))) {
    f14 = f13;
    i17 = i16;
    i18 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(32u)), 16777216u); k2 = k2 + 1u) {
    if ((f14 > pc[((k2) & 63u)])) {
    f15 = pc[((k2) & 63u)];
    i19 = pua[((k2) & 63u)];
    pc[((k2) & 63u)] = f14;
    pua[((k2) & 63u)] = i17;
    f14 = f15;
    i17 = i19;
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    cands[((((wid.x * 64u) + 32u) + k))] = (pua[((k) & 63u)]);
    }
    }
}
