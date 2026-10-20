// grid 256 1 1 1
// entry ingest grid 256 1 1 1
// entry search grid 64 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 256>;
var<workgroup> wg_c: array<f32, 512>;
var<workgroup> wg_d: array<f32, 32>;
var<workgroup> wg_e: array<f32, 64>;
var<workgroup> wg_f: array<f32, 512>;
var<workgroup> wg_ua: array<u32, 256>;
var<workgroup> wg_ub: array<u32, 32>;
var<workgroup> wg_uc: array<u32, 32>;
var<workgroup> wg_ud: array<u32, 16>;
var<workgroup> wg_ue: array<u32, 16>;
var<workgroup> wg_uf: array<u32, 17>;
var<workgroup> wg_ug: array<u32, 16>;
var<workgroup> wg_uh: array<u32, 512>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
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
    var f16r: f32 = 0.0;
    var f17: f32 = 0.0;
    var f18: f32 = 0.0;
    var f19: f32 = 0.0;
    var f20: f32 = 0.0;
    var f21: f32 = 0.0;
    var f22: f32 = 0.0;
    var f23: f32 = 0.0;
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
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    if ((cfg.pass_index) < (1u)) {
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = keys[((((((wid.x) & (8191u)) * 2048u) + (lid.x * 8u)) + k))];
    pa[((k) & 63u)] = f0;
    wg_a[((((lid.x * 8u) + k)) & 2047u)] = f0;
    f1 = (f1 + (f0 * f0));
    f2 = max(f2, abs(f0));
    }
    wg_b[((lid.x) & 255u)] = f1;
    wg_ua[((lid.x) & 255u)] = u32(bitcast<u32>(f2));
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    wg_b[((lid.x) & 255u)] = (wg_b[((lid.x) & 255u)] + wg_b[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    wg_ua[((lid.x) & 255u)] = max(wg_ua[((lid.x) & 255u)], wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    }
    workgroupBarrier();
    }
    f3 = wg_b[0u];
    f4 = bitcast<f32>(u32(wg_ua[0u]));
    if ((f3 > 0.0f)) {
    f5 = inverseSqrt(f3);
    }
    if ((f4 > 0.0f)) {
    f6 = (7.0f / f4);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    i0 = ((i0) | (u32(((u32(u32((i32(clamp(round((pa[((k) & 63u)] * f6)), -7.0f, 7.0f)) + 8i))) & 15u) << ((u32((4u * k))) & 31u)))));
    }
    index_f32[(((0u + (min((cfg.arrived_begin + wid.x), 24999u) * 256u)) + lid.x))] = (bitcast<f32>(u32(i0)));
    if ((lid.x == 0u)) {
    index_f32[((6662144u + (((cfg.arrived_begin + wid.x)) & (32767u))))] = (((f4 * f5) / 7.0f));
    }
    workgroupBarrier();
    for (k = 0u; k < 4u; k = k + 1u) {
    i1 = 0u;
    i1 = (((lid.x * 4u) + k) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    f7 = select(-1.0f, 1.0f, (((i1) & (16u)) != 0u));
    f8 = (f8 + (f7 * wg_a[((((((lid.x * 4u) + k) * 1237u) + 291u)) & 2047u)]));
    i1 = 0u;
    i1 = ((((lid.x + 256u) * 4u) + k) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    f9 = select(-1.0f, 1.0f, (((i1) & (16u)) != 0u));
    f10 = (f10 + (f9 * wg_a[(((((((lid.x + 256u) * 4u) + k) * 1237u) + 291u)) & 2047u)]));
    }
    wg_b[((lid.x) & 255u)] = ((f8 * f8) + (f10 * f10));
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    wg_b[((lid.x) & 255u)] = (wg_b[((lid.x) & 255u)] + wg_b[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    }
    workgroupBarrier();
    }
    f11 = wg_b[0u];
    if ((f11 > 0.0f)) {
    f12 = inverseSqrt(f11);
    }
    if ((((wid.x % max((select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end)) / 64u), 1u)) == 0u) & ((wid.x / max((select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end)) / 64u), 1u)) < 64u))) {
    index_f32[(((6400000u + (((((select((cfg.stage) - (1u), 0u, (1u) > (cfg.stage)) * 64u) + (wid.x / max((select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end)) / 64u), 1u)))) & (511u)) * 512u)) + lid.x))] = ((f8 * f12));
    index_f32[((((6400000u + (((((select((cfg.stage) - (1u), 0u, (1u) > (cfg.stage)) * 64u) + (wid.x / max((select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end)) / 64u), 1u)))) & (511u)) * 512u)) + lid.x) + 256u))] = ((f10 * f12));
    }
    } else {
    for (k = 0u; k < 4u; k = k + 1u) {
    i1 = 0u;
    i1 = (((lid.x * 4u) + k) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    f13 = select(-1.0f, 1.0f, (((i1) & (16u)) != 0u));
    f14 = (f14 + (f13 * keys[(((((wid.x) & (8191u)) * 2048u) + ((((((lid.x * 4u) + k) * 1237u) + 291u)) & (2047u))))]));
    i1 = 0u;
    i1 = ((((lid.x + 256u) * 4u) + k) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((15u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((12u) & 31u))));
    f15 = select(-1.0f, 1.0f, (((i1) & (16u)) != 0u));
    f16r = (f16r + (f15 * keys[(((((wid.x) & (8191u)) * 2048u) + (((((((lid.x + 256u) * 4u) + k) * 1237u) + 291u)) & (2047u))))]));
    }
    wg_a[(((1024u + lid.x)) & 2047u)] = f14;
    wg_a[(((1280u + lid.x)) & 2047u)] = f16r;
    for (k = 0u; k < 2u; k = k + 1u) {
    f17 = -3e38f;
    if ((((wid.x < min((cfg.stage * 64u), 512u)) & ((lid.x + (256u * k)) < min((cfg.stage * 64u), 512u))) & ((lid.x + (256u * k)) != wid.x))) {
    f18 = 0.0f;
    i2 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    f18 = (f18 + (index_f32[(((6400000u + (((wid.x) & (511u)) * 512u)) + k2))] * index_f32[(((6400000u + ((((lid.x + (256u * k))) & (511u)) * 512u)) + k2))]));
    }
    f17 = f18;
    }
    wg_a[(((lid.x + (256u * k))) & 2047u)] = f17;
    }
    workgroupBarrier();
    for (k = 0u; k < 2u; k = k + 1u) {
    f19 = -3e38f;
    if (((lid.x + (256u * k)) < min((cfg.stage * 64u), 512u))) {
    f20 = 0.0f;
    i3 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    f20 = (f20 + (wg_a[(((1024u + k2)) & 2047u)] * index_f32[(((6400000u + ((((lid.x + (256u * k))) & (511u)) * 512u)) + k2))]));
    }
    f19 = f20;
    }
    wg_a[(((512u + (lid.x + (256u * k)))) & 2047u)] = f19;
    }
    workgroupBarrier();
    for (k = 0u; k < 4u; k = k + 1u) {
    pi[((k) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (k = 0u; k < 20u; k = k + 1u) {
    f21 = wg_a[(((select(512u, 0u, (k < 16u)) + lid.x)) & 2047u)];
    i4 = lid.x;
    f22 = wg_a[((((select(512u, 0u, (k < 16u)) + lid.x) + 256u)) & 2047u)];
    if ((f22 > f21)) {
    f21 = f22;
    i4 = (lid.x + 256u);
    }
    wg_b[((lid.x) & 255u)] = f21;
    wg_ua[((lid.x) & 255u)] = i4;
    workgroupBarrier();
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    if ((lid.x < ((128u) >> ((k2) & 31u)))) {
    f23 = wg_b[(((lid.x + ((128u) >> ((k2) & 31u)))) & 255u)];
    i5 = wg_ua[(((lid.x + ((128u) >> ((k2) & 31u)))) & 255u)];
    if (((f23 > wg_b[((lid.x) & 255u)]) | ((f23 == wg_b[((lid.x) & 255u)]) & (i5 < wg_ua[((lid.x) & 255u)])))) {
    wg_b[((lid.x) & 255u)] = f23;
    wg_ua[((lid.x) & 255u)] = i5;
    }
    }
    workgroupBarrier();
    }
    if ((lid.x == 0u)) {
    i6 = ((wg_ua[0u]) & (511u));
    i7 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if ((wg_b[0u] > -1e38f)) {
    i7 = i6;
    wg_a[(((select(512u, 0u, (k < 16u)) + i6)) & 2047u)] = -3e38f;
    }
    if ((k < 16u)) {
    if ((wid.x < min((cfg.stage * 64u), 512u))) {
    atomicStore(&index_u32[(((131584u + (((wid.x) & (511u)) * 16u)) + k))], (i7));
    }
    } else {
    pi[((((select((k) - (16u), 0u, (16u) > (k))) & (3u))) & 15u)] = i7;
    }
    }
    workgroupBarrier();
    }
    if ((lid.x == 0u)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((!b0)) {
    i8 = pi[((k) & 15u)];
    if ((i8 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    i9 = ((i8) & (511u));
    i10 = atomicAdd(&index_u32[((0u + i9))], (1u));
    if ((i10 < 256u)) {
    atomicStore(&index_u32[(((512u + (i9 * 256u)) + i10))], ((cfg.arrived_begin + wid.x)));
    b0 = true;
    }
    }
    }
    }
    }
    }
    return;
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
    var f8: f32 = 0.0;
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
    var i20: u32 = 0u;
    var i21: u32 = 0u;
    var i22: u32 = 0u;
    var i23: u32 = 0u;
    var i24: u32 = 0u;
    var i25: u32 = 0u;
    var i26: u32 = 0u;
    var i27: u32 = 0u;
    var i28: u32 = 0u;
    var i29: u32 = 0u;
    var i30: u32 = 0u;
    var i31: u32 = 0u;
    var i32r: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
    wg_va[(((lid.x + (64u * k))) & 511u)] = queries[((((((wid.x) & (4095u)) * 512u) + lid.x) + (64u * k)))];
    }
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i0 = 0u;
    i0 = ((((lid.x + (64u * k)) * 4u) + k2) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))));
    i0 = u32((u32(i0) ^ (u32(i0) >> ((15u) & 31u))));
    i0 = (i0 * ((u32((lit_u32(11291u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(15469u))));
    i0 = u32((u32(i0) ^ (u32(i0) >> ((12u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i0) & (16u)) != 0u));
    f0 = (f0 + (f1 * wg_va[(((((((((((lid.x + (64u * k)) * 4u) + k2) * 1237u) + 291u)) & (2047u))) >> ((2u) & 31u))) & 511u)][((((((((lid.x + (64u * k)) * 4u) + k2) * 1237u) + 291u)) & (2047u))) & 3u]));
    }
    wg_c[(((lid.x + (64u * k))) & 511u)] = f0;
    }
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(32u)), 16777216u); k = k + 1u) {
    wg_ub[((k) & 31u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    wg_d[((k) & 31u)] = -3e38f;
    wg_uc[((k) & 31u)] = 1u;
    }
    for (k = 0u; k < 16u; k = k + 1u) {
    wg_ud[((k) & 15u)] = 0u;
    }
    for (k = 0u; k < 16u; k = k + 1u) {
    wg_ue[((k) & 15u)] = ((((k * min((cfg.stage * 64u), 512u)) / 16u)) & (511u));
    wg_ud[((((((((k * min((cfg.stage * 64u), 512u)) / 16u)) & (511u))) >> ((5u) & 31u))) & 15u)] = ((wg_ud[((((((((k * min((cfg.stage * 64u), 512u)) / 16u)) & (511u))) >> ((5u) & 31u))) & 15u)]) | (u32((1u << ((u32(((((k * min((cfg.stage * 64u), 512u)) / 16u)) & (511u)))) & 31u)))));
    }
    }
    workgroupBarrier();
    for (k = 0u; k < 25u; k = k + 1u) {
    i1 = wg_ue[((((lid.x) >> ((2u) & 31u))) & 15u)];
    f2 = 0.0f;
    if ((i1 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    i2 = ((6400000u + (((i1) & (511u)) * 512u)) + (((lid.x) & (3u)) * 128u));
    i3 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(128u)), 16777216u); k2 = k2 + 1u) {
    f2 = (f2 + (wg_c[((((((lid.x) & (3u)) * 128u) + k2)) & 511u)] * index_f32[((i2 + k2))]));
    }
    }
    wg_e[((lid.x) & 63u)] = f2;
    workgroupBarrier();
    if ((lid.x == 0u)) {
    i4 = 0u;
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    i5 = wg_ue[((k2) & 15u)];
    if ((i5 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    f3 = (((wg_e[(((k2 * 4u)) & 63u)] + wg_e[((((k2 * 4u) + 1u)) & 63u)]) + wg_e[((((k2 * 4u) + 2u)) & 63u)]) + wg_e[((((k2 * 4u) + 3u)) & 63u)]);
    i6 = i5;
    i7 = 0u;
    if ((f3 > wg_d[31u])) {
    i8 = 0u;
    for (k3 = 0u; k3 < min((lit_u32(32u)), 16777216u); k3 = k3 + 1u) {
    if ((f3 > wg_d[((k3) & 31u)])) {
    f4 = wg_d[((k3) & 31u)];
    i9 = wg_ub[((k3) & 31u)];
    i10 = wg_uc[((k3) & 31u)];
    wg_d[((k3) & 31u)] = f3;
    wg_ub[((k3) & 31u)] = i6;
    wg_uc[((k3) & 31u)] = i7;
    f3 = f4;
    i6 = i9;
    i7 = i10;
    }
    }
    }
    }
    }
    i11 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    i12 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(32u)), 16777216u); k2 = k2 + 1u) {
    if ((((i11 == ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & (wg_uc[((k2) & 31u)] == 0u)) & (wg_ub[((k2) & 31u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))))) {
    i11 = wg_ub[((k2) & 31u)];
    wg_uc[((k2) & 31u)] = 1u;
    }
    }
    for (k2 = 0u; k2 < min((lit_u32(16u)), 16777216u); k2 = k2 + 1u) {
    i13 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if ((i11 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    i14 = atomicLoad(&index_u32[(((131584u + (((i11) & (511u)) * 16u)) + k2))]);
    if (((i14 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & (i14 < min((cfg.stage * 64u), 512u)))) {
    i15 = ((((i14) >> ((5u) & 31u))) & (15u));
    i16 = u32((1u << ((u32(i14)) & 31u)));
    if ((((wg_ud[((i15) & 15u)]) & (i16)) == 0u)) {
    wg_ud[((i15) & 15u)] = ((wg_ud[((i15) & 15u)]) | (i16));
    i13 = i14;
    }
    }
    }
    wg_ue[((k2) & 15u)] = i13;
    }
    }
    workgroupBarrier();
    }
    if ((lid.x == 0u)) {
    for (k = 0u; k < min((lit_u32(12u)), 16777216u); k = k + 1u) {
    wg_uf[min((k), 16u)] = i17;
    i18 = wg_ub[((k) & 31u)];
    i19 = 0u;
    if ((i18 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    i19 = min(atomicLoad(&index_u32[((0u + ((i18) & (511u))))]), 256u);
    }
    wg_ug[((k) & 15u)] = ((i18) & (511u));
    i17 = (i17 + i19);
    }
    wg_uf[12u] = i17;
    }
    workgroupBarrier();
    i20 = wg_uf[12u];
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    pa[((k) & 63u)] = -3e38f;
    pi[((k) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (t = 0u; t < min((lit_u32(3072u)), 16777216u); t = t + 64u) {
    if (((t + lid.x) < i20)) {
    i21 = 0u;
    i22 = 1u;
    for (k = 0u; k < min((select((lit_u32(12u)) - (lit_u32(1u)), 0u, (lit_u32(1u)) > (lit_u32(12u)))), 16777216u); k = k + 1u) {
    if (((t + lid.x) >= wg_uf[min(((1u + k)), 16u)])) {
    i21 = (1u + k);
    }
    }
    i23 = atomicLoad(&index_u32[(((512u + (wg_ug[((i21) & 15u)] * 256u)) + ((select(((t + lid.x)) - (wg_uf[min((i21), 16u)]), 0u, (wg_uf[min((i21), 16u)]) > ((t + lid.x)))) & (255u))))]);
    if ((i23 < cfg.arrived_end)) {
    i24 = (0u + (min(i23, 24999u) * 256u));
    f5 = 0.0f;
    i25 = 0u;
    for (k = 0u; k < min((lit_u32(256u)), 16777216u); k = k + 1u) {
    i26 = u32(bitcast<u32>(index_f32[((i24 + k))]));
    f5 = ((f5 + dot((vec4<f32>(vec3<f32>(vec2<f32>(f32(((i26) & (15u))), f32(((((i26) >> ((4u) & 31u))) & (15u)))), f32(((((i26) >> ((8u) & 31u))) & (15u)))), f32(((((i26) >> ((12u) & 31u))) & (15u)))) - vec4<f32>(8.0f)), wg_va[(((k * 2u)) & 511u)])) + dot((vec4<f32>(vec3<f32>(vec2<f32>(f32(((((i26) >> ((16u) & 31u))) & (15u))), f32(((((i26) >> ((20u) & 31u))) & (15u)))), f32(((((i26) >> ((24u) & 31u))) & (15u)))), f32(((((i26) >> ((28u) & 31u))) & (15u)))) - vec4<f32>(8.0f)), wg_va[((((k * 2u) + 1u)) & 511u)]));
    }
    f6 = (f5 * index_f32[((6662144u + ((i23) & (32767u))))]);
    i27 = i23;
    if ((f6 > pa[7u])) {
    i28 = 0u;
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    if ((f6 > pa[((k) & 63u)])) {
    f7 = pa[((k) & 63u)];
    i29 = pi[((k) & 15u)];
    pa[((k) & 63u)] = f6;
    pi[((k) & 15u)] = i27;
    f6 = f7;
    i27 = i29;
    }
    }
    }
    }
    }
    }
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    wg_f[((((lid.x * 8u) + k)) & 511u)] = pa[((k) & 63u)];
    wg_uh[((((lid.x * 8u) + k)) & 511u)] = pi[((k) & 15u)];
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    i30 = wg_uh[((((lid.x * 8u) + k)) & 511u)];
    if ((i30 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    f8 = wg_f[((((lid.x * 8u) + k)) & 511u)];
    i31 = 0u;
    i32r = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    if (((wg_uh[((k2) & 511u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & ((wg_f[((k2) & 511u)] > f8) | ((wg_f[((k2) & 511u)] == f8) & (k2 < ((lid.x * 8u) + k)))))) {
    i31 = (i31 + 1u);
    }
    }
    if ((i31 < 64u)) {
    cands[(((wid.x * 64u) + i31))] = (i30);
    }
    }
    }
    }
}
