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
var<workgroup> wg_a: array<f32, 64>;
var<workgroup> wg_b: array<f32, 64>;
var<workgroup> wg_ua: array<u32, 64>;
var<workgroup> wg_ub: array<u32, 64>;
fn lit_u32(x: u32) -> u32 { return x; }
fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { var t = v; return t[i & 3u]; }
@compute @workgroup_size(1, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var k: u32 = 0u;
    if ((((cfg.arrived_begin + wid.x) < cfg.arrived_end) & ((cfg.arrived_begin + wid.x) < 25000u))) {
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    index_f32[((((cfg.arrived_begin + wid.x) * 64u) + k))] = (keys[(((select(((cfg.arrived_begin + wid.x)) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > ((cfg.arrived_begin + wid.x))) * 512u) + (k * 8u)))].x);
    }
    }
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
    var f9: f32 = 0.0;
    var f10: f32 = 0.0;
    var f11: f32 = 0.0;
    var f12: f32 = 0.0;
    var f13: f32 = 0.0;
    var f14: f32 = 0.0;
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
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    var pua: array<u32, 64>;
    for (t = 0u; t < min((lit_u32(4u)), 16777216u); t = t + 1u) {
    pa[((t) & 63u)] = -3e38f;
    pi[((t) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    pb[((t) & 63u)] = -3e38f;
    pj[((t) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(64u)), 16777216u); k = k + 1u) {
    f0 = (f0 + (at_vec4f(queries[(((wid.x * 512u) + (k * 8u)))], 0u) * at_vec4f(queries[(((wid.x * 512u) + (k * 8u)))], 0u)));
    }
    for (k = 0u; k < min((cfg.arrived_end), 16777216u); k = k + 16u) {
    if ((((k + lid.x) < cfg.arrived_end) & ((k + lid.x) < 25000u))) {
    f1 = 0.0f;
    f2 = 0.0f;
    i0 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(64u)), 16777216u); k2 = k2 + 1u) {
    f3 = index_f32[((((k + lid.x) * 64u) + k2))];
    f1 = (f1 + (at_vec4f(queries[(((wid.x * 512u) + (k2 * 8u)))], 0u) * f3));
    f2 = (f2 + (f3 * f3));
    }
    f4 = -3e38f;
    if ((sqrt((f0 * f2)) > 0.0f)) {
    f4 = (f1 / sqrt((f0 * f2)));
    }
    if (((k + lid.x) >= cfg.arrived_begin)) {
    if ((f4 > pa[3u])) {
    f5 = f4;
    i1 = (k + lid.x);
    for (k2 = 0u; k2 < min((lit_u32(4u)), 16777216u); k2 = k2 + 1u) {
    if ((f5 > pa[((k2) & 63u)])) {
    f6 = pa[((k2) & 63u)];
    i2 = pi[((k2) & 15u)];
    pa[((k2) & 63u)] = f5;
    pi[((k2) & 15u)] = i1;
    f5 = f6;
    i1 = i2;
    }
    }
    }
    } else {
    if ((f4 > pb[3u])) {
    f7 = f4;
    i3 = (k + lid.x);
    for (k2 = 0u; k2 < min((lit_u32(4u)), 16777216u); k2 = k2 + 1u) {
    if ((f7 > pb[((k2) & 63u)])) {
    f8 = pb[((k2) & 63u)];
    i4 = pj[((k2) & 15u)];
    pb[((k2) & 63u)] = f7;
    pj[((k2) & 15u)] = i3;
    f7 = f8;
    i3 = i4;
    }
    }
    }
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(4u)), 16777216u); k = k + 1u) {
    wg_a[((((lid.x * 4u) + k)) & 63u)] = pa[((k) & 63u)];
    wg_ua[((((lid.x * 4u) + k)) & 63u)] = pi[((k) & 15u)];
    wg_b[((((lid.x * 4u) + k)) & 63u)] = pb[((k) & 63u)];
    wg_ub[((((lid.x * 4u) + k)) & 63u)] = pj[((k) & 15u)];
    }
    workgroupBarrier();
    if (((wid.x < cfg.n_queries) & (lid.x == 0u))) {
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    pc[((k) & 63u)] = -3e38f;
    pua[((k) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < 64u; k = k + 1u) {
    f9 = wg_a[((k) & 63u)];
    i5 = wg_ua[((k) & 63u)];
    if (((i5 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & (f9 > pc[31u]))) {
    f10 = f9;
    i6 = i5;
    for (k2 = 0u; k2 < min((lit_u32(32u)), 16777216u); k2 = k2 + 1u) {
    if ((f10 > pc[((k2) & 63u)])) {
    f11 = pc[((k2) & 63u)];
    i7 = pua[((k2) & 63u)];
    pc[((k2) & 63u)] = f10;
    pua[((k2) & 63u)] = i6;
    f10 = f11;
    i6 = i7;
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    cands[(((wid.x * 64u) + k))] = (pua[((k) & 63u)]);
    }
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    pc[((k) & 63u)] = -3e38f;
    pua[((k) & 63u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < 64u; k = k + 1u) {
    f12 = wg_b[((k) & 63u)];
    i8 = wg_ub[((k) & 63u)];
    if (((i8 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & (f12 > pc[31u]))) {
    f13 = f12;
    i9 = i8;
    for (k2 = 0u; k2 < min((lit_u32(32u)), 16777216u); k2 = k2 + 1u) {
    if ((f13 > pc[((k2) & 63u)])) {
    f14 = pc[((k2) & 63u)];
    i10 = pua[((k2) & 63u)];
    pc[((k2) & 63u)] = f13;
    pua[((k2) & 63u)] = i9;
    f13 = f14;
    i9 = i10;
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    cands[((((wid.x * 64u) + 32u) + k))] = (pua[((k) & 63u)]);
    }
    }
}
