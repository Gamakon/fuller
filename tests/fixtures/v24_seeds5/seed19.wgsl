// grid 256 1 1 1
// entry ingest grid 256 1 1 1
// entry search grid 256 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_au_a: array<atomic<u32>, 40>;
var<workgroup> wg_b: array<f32, 256>;
var<workgroup> wg_ua: array<u32, 8>;
var<workgroup> wg_ub: array<u32, 16>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(256, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
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
    var i12: u32 = 0u;
    var i13: u32 = 0u;
    var i14: u32 = 0u;
    var i15: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    if ((cfg.pass_index) < (1u)) {
    for (t = 0u; t < min((lit_u32(512u)), 16777216u); t = t + 256u) {
    wg_va[(((t + lid.x)) & 511u)] = keys[(((((wid.x) & (8191u)) * 512u) + (t + lid.x)))];
    }
    workgroupBarrier();
    i0 = 0u;
    f1 = 0.0f;
    i1 = 0u;
    for (k = 0u; k < 64u; k = k + 1u) {
    i3 = 0u;
    i4 = u32(((u32((lid.x * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31161u))))) ^ u32(((k + 1u) * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u)))))) ^ ((lit_u32(9541u) << ((lit_u32(16u)) & 31u)) | lit_u32(62609u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((16u) & 31u))));
    i4 = (i4 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((15u) & 31u))));
    i4 = (i4 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((16u) & 31u))));
    i2 = i4;
    i0 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    f1 = (f1 + dot(wg_va[((((k * 8u) + k2)) & 511u)], ((vec4<f32>(vec3<f32>(vec2<f32>(f32(((((i2) >> (((k2 * 4u)) & 31u))) & (1u))), f32(((((((i2) >> (((k2 * 4u)) & 31u))) >> ((1u) & 31u))) & (1u)))), f32(((((((i2) >> (((k2 * 4u)) & 31u))) >> ((2u) & 31u))) & (1u)))), f32(((((((i2) >> (((k2 * 4u)) & 31u))) >> ((3u) & 31u))) & (1u)))) * vec4<f32>(2.0f)) - vec4<f32>(1.0f))));
    }
    }
    f0 = f1;
    wg_a[((lid.x) & 255u)] = f0;
    wg_b[((lid.x) & 255u)] = (f0 * f0);
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    wg_b[((lid.x) & 255u)] = (wg_b[((lid.x) & 255u)] + wg_b[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    }
    workgroupBarrier();
    }
    if ((wid.x < select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end)))) {
    if ((lid.x < 128u)) {
    index_u32[(((((0u + ((((cfg.arrived_begin + wid.x)) & (32767u)) * 128u)) + lid.x)) & (8388607u)))] = (u32(pack2x16float(vec2<f32>((wg_a[(((2u * lid.x)) & 255u)] * select(0.0f, (1.0f / sqrt(wg_b[0u])), (sqrt(wg_b[0u]) > 0.0f))), (wg_a[((((2u * lid.x) + 1u)) & 255u)] * select(0.0f, (1.0f / sqrt(wg_b[0u])), (sqrt(wg_b[0u]) > 0.0f)))))));
    }
    if ((lid.x < 8u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    if ((wg_a[((((lid.x * 32u) + k)) & 255u)] > 0.0f)) {
    i5 = ((i5) | (((((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u)))) >> ((k) & 31u))));
    }
    }
    index_u32[(((4194304u + (lid.x * 32768u)) + (((cfg.arrived_begin + wid.x)) & (32767u))))] = (i5);
    }
    }
    }
    if ((cfg.pass_index) == (1u)) {
    if ((lid.x < 8u)) {
    atomicStore(&wg_au_a[min((lid.x), 39u)], 0u);
    }
    workgroupBarrier();
    for (t = 0u; t < min((select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))), 16777216u); t = t + 32u) {
    if (((t + ((lid.x) >> ((3u) & 31u))) < select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end)))) {
    if (((index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + (((cfg.arrived_begin + (t + ((lid.x) >> ((3u) & 31u))))) & (32767u))))] < index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + (((cfg.arrived_begin + wid.x)) & (32767u))))]) | ((index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + (((cfg.arrived_begin + (t + ((lid.x) >> ((3u) & 31u))))) & (32767u))))] == index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + (((cfg.arrived_begin + wid.x)) & (32767u))))]) & ((cfg.arrived_begin + (t + ((lid.x) >> ((3u) & 31u)))) < (cfg.arrived_begin + wid.x))))) {
    i6 = (i6 + 1u);
    }
    }
    }
    _ = atomicAdd(&wg_au_a[min((((lid.x) & (7u))), 39u)], i6);
    workgroupBarrier();
    if (((lid.x < 8u) & (wid.x < select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))))) {
    index_u32[(((5242880u + (lid.x * 8192u)) + ((atomicLoad(&wg_au_a[min((lid.x), 39u)])) & (8191u))))] = ((cfg.arrived_begin + wid.x));
    }
    }
    if (((cfg.pass_index == 2u) & (wid.x < select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))))) {
    if ((((lid.x) >> ((3u) & 31u)) == 0u)) {
    i7 = index_u32[(((5242880u + (((lid.x) & (7u)) * 8192u)) + ((wid.x) & (8191u))))];
    i8 = cfg.arrived_begin;
    for (k = 0u; k < 16u; k = k + 1u) {
    if ((i9 < i8)) {
    i10 = index_u32[(((4456448u + (((((u32(((u32(cfg.stage) & 1u) ^ 1u))) & (1u)) * 8u) + ((((lid.x) & (7u))) & (7u))) * 32768u)) + (((((i9 + i8)) >> ((1u) & 31u))) & (32767u))))];
    if (((index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i10) & (32767u))))] < index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i7) & (32767u))))]) | ((index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i10) & (32767u))))] == index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i7) & (32767u))))]) & (i10 < i7)))) {
    i9 = ((((i9 + i8)) >> ((1u) & 31u)) + 1u);
    } else {
    i8 = (((i9 + i8)) >> ((1u) & 31u));
    }
    }
    }
    index_u32[(((4456448u + (((((((cfg.stage) & (1u))) & (1u)) * 8u) + ((((lid.x) & (7u))) & (7u))) * 32768u)) + (((i9 + wid.x)) & (32767u))))] = (i7);
    index_u32[(((4980736u + (((lid.x) & (7u)) * 32768u)) + ((i7) & (32767u))))] = ((i9 + wid.x));
    } else {
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((((select((((k * 31u) + ((lid.x) >> ((3u) & 31u)))) - (1u), 0u, (1u) > (((k * 31u) + ((lid.x) >> ((3u) & 31u))))) * select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))) + wid.x) < cfg.arrived_begin)) {
    i11 = index_u32[(((4456448u + (((((u32(((u32(cfg.stage) & 1u) ^ 1u))) & (1u)) * 8u) + ((((lid.x) & (7u))) & (7u))) * 32768u)) + ((((select((((k * 31u) + ((lid.x) >> ((3u) & 31u)))) - (1u), 0u, (1u) > (((k * 31u) + ((lid.x) >> ((3u) & 31u))))) * select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))) + wid.x)) & (32767u))))];
    i12 = 0u;
    i13 = select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end));
    i14 = 0u;
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    if ((i12 < i13)) {
    i15 = index_u32[(((5242880u + (((lid.x) & (7u)) * 8192u)) + (((((i12 + i13)) >> ((1u) & 31u))) & (8191u))))];
    if (((index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i15) & (32767u))))] < index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i11) & (32767u))))]) | ((index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i15) & (32767u))))] == index_u32[(((4194304u + (((((lid.x) & (7u))) & (7u)) * 32768u)) + ((i11) & (32767u))))]) & (i15 < i11)))) {
    i12 = ((((i12 + i13)) >> ((1u) & 31u)) + 1u);
    } else {
    i13 = (((i12 + i13)) >> ((1u) & 31u));
    }
    }
    }
    index_u32[(((4456448u + (((((((cfg.stage) & (1u))) & (1u)) * 8u) + ((((lid.x) & (7u))) & (7u))) * 32768u)) + (((((select((((k * 31u) + ((lid.x) >> ((3u) & 31u)))) - (1u), 0u, (1u) > (((k * 31u) + ((lid.x) >> ((3u) & 31u))))) * select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))) + wid.x) + i12)) & (32767u))))] = (i11);
    index_u32[(((4980736u + (((lid.x) & (7u)) * 32768u)) + ((i11) & (32767u))))] = ((((select((((k * 31u) + ((lid.x) >> ((3u) & 31u)))) - (1u), 0u, (1u) > (((k * 31u) + ((lid.x) >> ((3u) & 31u))))) * select((cfg.arrived_end) - (cfg.arrived_begin), 0u, (cfg.arrived_begin) > (cfg.arrived_end))) + wid.x) + i12));
    }
    }
    }
    }
}
@compute @workgroup_size(256, 1, 1)
fn search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
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
    var b0: bool = false;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    if ((lid.x < 40u)) {
    atomicStore(&wg_au_a[min((lid.x), 39u)], 0u);
    }
    for (t = 0u; t < min((lit_u32(512u)), 16777216u); t = t + 256u) {
    wg_va[(((t + lid.x)) & 511u)] = queries[(((((wid.x) & (4095u)) * 512u) + (t + lid.x)))];
    }
    workgroupBarrier();
    i0 = 0u;
    f1 = 0.0f;
    i1 = 0u;
    for (k = 0u; k < 64u; k = k + 1u) {
    i3 = 0u;
    i4 = u32(((u32((lid.x * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31161u))))) ^ u32(((k + 1u) * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u)))))) ^ ((lit_u32(9541u) << ((lit_u32(16u)) & 31u)) | lit_u32(62609u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((16u) & 31u))));
    i4 = (i4 * ((u32((lit_u32(32747u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(13613u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((15u) & 31u))));
    i4 = (i4 * ((u32((lit_u32(33900u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(42635u))));
    i4 = u32((u32(i4) ^ (u32(i4) >> ((16u) & 31u))));
    i2 = i4;
    i0 = 0u;
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    f1 = (f1 + dot(wg_va[((((k * 8u) + k2)) & 511u)], ((vec4<f32>(vec3<f32>(vec2<f32>(f32(((((i2) >> (((k2 * 4u)) & 31u))) & (1u))), f32(((((((i2) >> (((k2 * 4u)) & 31u))) >> ((1u) & 31u))) & (1u)))), f32(((((((i2) >> (((k2 * 4u)) & 31u))) >> ((2u) & 31u))) & (1u)))), f32(((((((i2) >> (((k2 * 4u)) & 31u))) >> ((3u) & 31u))) & (1u)))) * vec4<f32>(2.0f)) - vec4<f32>(1.0f))));
    }
    }
    f0 = f1;
    wg_a[((lid.x) & 255u)] = f0;
    workgroupBarrier();
    if ((lid.x < 8u)) {
    for (k = 0u; k < 32u; k = k + 1u) {
    if ((wg_a[((((lid.x * 32u) + k)) & 255u)] > 0.0f)) {
    i5 = ((i5) | (((((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u)))) >> ((k) & 31u))));
    }
    }
    wg_ua[((lid.x) & 7u)] = i5;
    }
    workgroupBarrier();
    if ((lid.x < 8u)) {
    i6 = wg_ua[((lid.x) & 7u)];
    i7 = cfg.arrived_end;
    for (k = 0u; k < 16u; k = k + 1u) {
    if ((i8 < i7)) {
    if ((index_u32[(((4194304u + (((lid.x) & (7u)) * 32768u)) + ((index_u32[(((4456448u + (((((((cfg.stage) & (1u))) & (1u)) * 8u) + ((lid.x) & (7u))) * 32768u)) + (((((i8 + i7)) >> ((1u) & 31u))) & (32767u))))]) & (32767u))))] < i6)) {
    i8 = ((((i8 + i7)) >> ((1u) & 31u)) + 1u);
    } else {
    i7 = (((i8 + i7)) >> ((1u) & 31u));
    }
    }
    }
    if ((cfg.arrived_end > 256u)) {
    i9 = select((i8) - (min(i8, 128u)), 0u, (min(i8, 128u)) > (i8));
    i9 = min(i9, select((cfg.arrived_end) - (256u), 0u, (256u) > (cfg.arrived_end)));
    }
    wg_ub[((lid.x) & 15u)] = i9;
    wg_ub[(((8u + lid.x)) & 15u)] = min(256u, cfg.arrived_end);
    }
    workgroupBarrier();
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    i10 = 0u;
    i11 = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    if ((lid.x < wg_ub[(((8u + t)) & 15u)])) {
    i12 = index_u32[(((4456448u + (((((((cfg.stage) & (1u))) & (1u)) * 8u) + ((t) & (7u))) * 32768u)) + (((wg_ub[((t) & 15u)] + lid.x)) & (32767u))))];
    b0 = false;
    i13 = 0u;
    for (k = 0u; k < min((lit_u32(8u)), 16777216u); k = k + 1u) {
    if ((k < t)) {
    if (((index_u32[(((4980736u + (k * 32768u)) + ((i12) & (32767u))))] >= wg_ub[((k) & 15u)]) & (index_u32[(((4980736u + (k * 32768u)) + ((i12) & (32767u))))] < (wg_ub[((k) & 15u)] + wg_ub[(((8u + k)) & 15u)])))) {
    b0 = true;
    }
    }
    }
    if (((!b0) & (i12 < cfg.arrived_end))) {
    f2 = 0.0f;
    i14 = (0u + (((i12) & (32767u)) * 128u));
    for (k = 0u; k < min((lit_u32(128u)), 16777216u); k = k + 1u) {
    f2 = ((f2 + (unpack2x16float(u32(index_u32[((((i14 + k)) & (8388607u)))])).x * wg_a[(((2u * k)) & 255u)])) + (unpack2x16float(u32(index_u32[((((i14 + k)) & (8388607u)))])).y * wg_a[((((2u * k) + 1u)) & 255u)]));
    }
    i10 = max(select(((u32(bitcast<u32>(f2))) | (((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u))))), u32((~bitcast<u32>(f2))), (((u32(bitcast<u32>(f2))) - ((u32(bitcast<u32>(f2))) % 2147483648u)) != 0u)), 1u);
    i11 = i12;
    }
    }
    pi[((t) & 15u)] = i10;
    pj[((t) & 15u)] = i11;
    }
    for (k = 0u; k < 32u; k = k + 1u) {
    i15 = 0u;
    i16 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(8u)), 16777216u); k2 = k2 + 1u) {
    if ((pi[((k2) & 15u)] >= ((i17) | (((((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u)))) >> ((k) & 31u)))))) {
    i15 = (i15 + 1u);
    }
    }
    if ((i15 > 0u)) {
    _ = atomicAdd(&wg_au_a[min((k), 39u)], i15);
    }
    workgroupBarrier();
    if ((atomicLoad(&wg_au_a[min((k), 39u)]) >= 64u)) {
    i17 = ((i17) | (((((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u)))) >> ((k) & 31u))));
    }
    }
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    if ((((wid.x < cfg.n_queries) & (pi[((t) & 15u)] > i17)) & (pj[((t) & 15u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))))) {
    i18 = atomicAdd(&wg_au_a[32u], 1u);
    if ((i18 < 64u)) {
    cands[(((wid.x * 64u) + i18))] = (pj[((t) & 15u)]);
    }
    }
    }
    workgroupBarrier();
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    if (((((wid.x < cfg.n_queries) & (i17 > 0u)) & (pi[((t) & 15u)] == i17)) & (pj[((t) & 15u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))))) {
    i19 = atomicAdd(&wg_au_a[32u], 1u);
    if ((i19 < 64u)) {
    cands[(((wid.x * 64u) + i19))] = (pj[((t) & 15u)]);
    }
    }
    }
    return;
}
