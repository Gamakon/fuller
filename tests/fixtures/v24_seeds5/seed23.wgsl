// grid 256 1 1 1
// entry ingest grid 256 1 1 1
// entry search grid 256 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
var<workgroup> wg_c: array<f32, 1024>;
var<workgroup> wg_ua: array<u32, 256>;
var<workgroup> wg_ub: array<u32, 1>;
var<workgroup> wg_uc: array<u32, 1024>;
var<workgroup> wg_va: array<vec4<f32>, 256>;
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
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
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
    var t: u32 = 0u;
    var k: u32 = 0u;
    if ((cfg.pass_index) < (1u)) {
    i0 = 0u;
    i1 = (((4u * lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f0 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = ((((4u * lid.x) + 1u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = ((((4u * lid.x) + 2u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f2 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = ((((4u * lid.x) + 3u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f3 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    i0 = 0u;
    i1 = ((((4u * lid.x) + 1024u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f0 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((4u * lid.x) + 1024u) + 1u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((4u * lid.x) + 1024u) + 2u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f2 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((4u * lid.x) + 1024u) + 3u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f3 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    v1 = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    v2 = ((v0 * keys[(((((wid.x) & (8191u)) * 512u) + lid.x))]) + (v1 * keys[((((((wid.x) & (8191u)) * 512u) + 256u) + lid.x))]));
    wg_a[((lid.x) & 255u)] = dot(v2, v2);
    wg_b[((lid.x) & 255u)] = max(max(abs(v2).x, abs(v2).y), max(abs(v2).z, abs(v2).w));
    workgroupBarrier();
    i2 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    wg_a[((lid.x) & 255u)] = (wg_a[((lid.x) & 255u)] + wg_a[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    wg_b[((lid.x) & 255u)] = max(wg_b[((lid.x) & 255u)], wg_b[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    }
    workgroupBarrier();
    }
    f4 = sqrt(wg_a[0u]);
    f5 = wg_b[0u];
    if (((f4 > 0.0f) & (f5 > 0.0f))) {
    v3 = (v2 / vec4<f32>(f4));
    f6 = (f5 / f4);
    i3 = u32(pack4x8snorm((v2 / vec4<f32>(f5))));
    }
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    atomicStore(&index_u32[(((0u + (min((cfg.arrived_begin + wid.x), 25599u) * 256u)) + lid.x))], (i3));
    if ((lid.x == 0u)) {
    index_f32[((262144u + min((cfg.arrived_begin + wid.x), 25599u)))] = (f6);
    }
    if (((cfg.stage == 1u) & (wid.x < 256u))) {
    index_f32[(((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)))] = (v3[(0u) & 3u]);
    index_f32[((((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 1u))] = (v3[(1u) & 3u]);
    index_f32[((((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 2u))] = (v3[(2u) & 3u]);
    index_f32[((((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 3u))] = (v3[(3u) & 3u]);
    }
    }
    } else {
    if ((cfg.pass_index) == (1u)) {
    v4 = (unpack4x8snorm(u32(atomicLoad(&index_u32[(((0u + (min((cfg.arrived_begin + wid.x), 25599u) * 256u)) + lid.x))]))) * vec4<f32>(index_f32[((262144u + min((cfg.arrived_begin + wid.x), 25599u)))]));
    wg_va[((lid.x) & 255u)] = v4;
    workgroupBarrier();
    f8 = 0.0f;
    i4 = 0u;
    for (t = 0u; t < min((lit_u32(256u)), 16777216u); t = t + 1u) {
    f8 = (f8 + dot(vec4<f32>(vec3<f32>(vec2<f32>(index_f32[(((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)))], index_f32[((((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)) + 1u))]), index_f32[((((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)) + 2u))]), index_f32[((((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)) + 3u))]), wg_va[((t) & 255u)]));
    }
    f7 = f8;
    f9 = f7;
    wg_a[((lid.x) & 255u)] = f9;
    wg_ua[((lid.x) & 255u)] = lid.x;
    workgroupBarrier();
    i5 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    f10 = wg_a[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)];
    if (((f10 > wg_a[((lid.x) & 255u)]) | ((f10 == wg_a[((lid.x) & 255u)]) & (wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)] < wg_ua[((lid.x) & 255u)])))) {
    wg_a[((lid.x) & 255u)] = f10;
    wg_ua[((lid.x) & 255u)] = wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)];
    }
    }
    workgroupBarrier();
    }
    i6 = ((wg_ua[0u]) & (255u));
    f11 = wg_a[0u];
    workgroupBarrier();
    if ((lid.x == i6)) {
    f9 = -3e38f;
    }
    wg_a[((lid.x) & 255u)] = f9;
    wg_ua[((lid.x) & 255u)] = lid.x;
    workgroupBarrier();
    i5 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    f10 = wg_a[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)];
    if (((f10 > wg_a[((lid.x) & 255u)]) | ((f10 == wg_a[((lid.x) & 255u)]) & (wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)] < wg_ua[((lid.x) & 255u)])))) {
    wg_a[((lid.x) & 255u)] = f10;
    wg_ua[((lid.x) & 255u)] = wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)];
    }
    }
    workgroupBarrier();
    }
    i7 = ((wg_ua[0u]) & (255u));
    if ((lid.x == 0u)) {
    i8 = i6;
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    i9 = atomicAdd(&index_u32[((6815744u + i6))], (1u));
    if ((i9 < 4096u)) {
    atomicStore(&index_u32[(((6841856u + (i6 * 4096u)) + i9))], ((cfg.arrived_begin + wid.x)));
    } else {
    i8 = i7;
    i10 = atomicAdd(&index_u32[((6815744u + i7))], (1u));
    if ((i10 < 4096u)) {
    atomicStore(&index_u32[(((6841856u + (i7 * 4096u)) + i10))], ((cfg.arrived_begin + wid.x)));
    }
    }
    atomicStore(&index_u32[((6816256u + min((cfg.arrived_begin + wid.x), 25599u)))], (i8));
    _ = atomicAdd(&index_u32[((6816000u + i8))], (1u));
    if ((((i8 == i6) & (i7 != i6)) & (wg_a[0u] >= (f11 - 0.05f)))) {
    i11 = atomicAdd(&index_u32[((6815744u + i7))], (1u));
    if ((i11 < 4096u)) {
    atomicStore(&index_u32[(((6841856u + (i7 * 4096u)) + i11))], ((((cfg.arrived_begin + wid.x)) | (((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u)))))));
    }
    }
    }
    wg_ub[0u] = i8;
    }
    workgroupBarrier();
    if (((cfg.arrived_begin + wid.x) < cfg.arrived_end)) {
    _ = atomicAdd(&index_u32[(((6553600u + (((wg_ub[0u]) & (255u)) * 1024u)) + (4u * lid.x)))], (u32(bitcast<u32>(i32(round((v4.x * 16384.0f)))))));
    _ = atomicAdd(&index_u32[((((6553600u + (((wg_ub[0u]) & (255u)) * 1024u)) + (4u * lid.x)) + 1u))], (u32(bitcast<u32>(i32(round((v4.y * 16384.0f)))))));
    _ = atomicAdd(&index_u32[((((6553600u + (((wg_ub[0u]) & (255u)) * 1024u)) + (4u * lid.x)) + 2u))], (u32(bitcast<u32>(i32(round((v4.z * 16384.0f)))))));
    _ = atomicAdd(&index_u32[((((6553600u + (((wg_ub[0u]) & (255u)) * 1024u)) + (4u * lid.x)) + 3u))], (u32(bitcast<u32>(i32(round((v4.w * 16384.0f)))))));
    }
    } else {
    v5 = vec4<f32>(vec3<f32>(vec2<f32>(f32(bitcast<i32>(u32(atomicLoad(&index_u32[(((6553600u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)))])))), f32(bitcast<i32>(u32(atomicLoad(&index_u32[((((6553600u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 1u))]))))), f32(bitcast<i32>(u32(atomicLoad(&index_u32[((((6553600u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 2u))]))))), f32(bitcast<i32>(u32(atomicLoad(&index_u32[((((6553600u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 3u))])))));
    wg_a[((lid.x) & 255u)] = dot(v5, v5);
    workgroupBarrier();
    i12 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    wg_a[((lid.x) & 255u)] = (wg_a[((lid.x) & 255u)] + wg_a[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    }
    workgroupBarrier();
    }
    f12 = sqrt(wg_a[0u]);
    if ((((wid.x < 256u) & (atomicLoad(&index_u32[((6816000u + ((wid.x) & (255u))))]) > 0u)) & (f12 > 0.0f))) {
    v6 = (v5 / vec4<f32>(f12));
    index_f32[(((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)))] = (v6.x);
    index_f32[((((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 1u))] = (v6.y);
    index_f32[((((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 2u))] = (v6.z);
    index_f32[((((0u + (((wid.x) & (255u)) * 1024u)) + (4u * lid.x)) + 3u))] = (v6.w);
    }
    }
    }
    return;
}
@compute @workgroup_size(256, 1, 1)
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
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
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
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    i0 = 0u;
    i1 = (((4u * lid.x) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f0 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = ((((4u * lid.x) + 1u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = ((((4u * lid.x) + 2u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f2 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = ((((4u * lid.x) + 3u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f3 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    i0 = 0u;
    i1 = ((((4u * lid.x) + 1024u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f0 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((4u * lid.x) + 1024u) + 1u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f1 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((4u * lid.x) + 1024u) + 2u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f2 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    i0 = 0u;
    i1 = (((((4u * lid.x) + 1024u) + 3u) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u)))) + ((u32((lit_u32(32586u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31765u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(34283u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(51819u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((13u) & 31u))));
    i1 = (i1 * ((u32((lit_u32(49842u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(44597u))));
    i1 = u32((u32(i1) ^ (u32(i1) >> ((16u) & 31u))));
    f3 = select(-1.0f, 1.0f, (((i1) & (1u)) == 1u));
    v1 = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    v2 = ((v0 * queries[(((min(wid.x, 4095u) * 512u) + lid.x))]) + (v1 * queries[((((min(wid.x, 4095u) * 512u) + 256u) + lid.x))]));
    wg_a[((lid.x) & 255u)] = dot(v2, v2);
    workgroupBarrier();
    i2 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    wg_a[((lid.x) & 255u)] = (wg_a[((lid.x) & 255u)] + wg_a[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)]);
    }
    workgroupBarrier();
    }
    f4 = sqrt(wg_a[0u]);
    if ((f4 > 0.0f)) {
    v3 = (v2 / vec4<f32>(f4));
    }
    workgroupBarrier();
    wg_va[((lid.x) & 255u)] = v3;
    workgroupBarrier();
    f6 = 0.0f;
    i3 = 0u;
    for (t = 0u; t < min((lit_u32(256u)), 16777216u); t = t + 1u) {
    f6 = (f6 + dot(vec4<f32>(vec3<f32>(vec2<f32>(index_f32[(((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)))], index_f32[((((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)) + 1u))]), index_f32[((((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)) + 2u))]), index_f32[((((0u + (((lid.x) & (255u)) * 1024u)) + (4u * t)) + 3u))]), wg_va[((t) & 255u)]));
    }
    f5 = f6;
    f7 = f5;
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    wg_a[((lid.x) & 255u)] = f7;
    wg_ua[((lid.x) & 255u)] = lid.x;
    workgroupBarrier();
    i4 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((lid.x < ((128u) >> ((k) & 31u)))) {
    f8 = wg_a[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)];
    if (((f8 > wg_a[((lid.x) & 255u)]) | ((f8 == wg_a[((lid.x) & 255u)]) & (wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)] < wg_ua[((lid.x) & 255u)])))) {
    wg_a[((lid.x) & 255u)] = f8;
    wg_ua[((lid.x) & 255u)] = wg_ua[(((lid.x + ((128u) >> ((k) & 31u)))) & 255u)];
    }
    }
    workgroupBarrier();
    }
    i5 = ((wg_ua[0u]) & (255u));
    workgroupBarrier();
    pi[((t) & 15u)] = i5;
    if ((lid.x == i5)) {
    f7 = -3e38f;
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    pa[((k) & 63u)] = -3e38f;
    pj[((k) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (t = 0u; t < min((lit_u32(8u)), 16777216u); t = t + 1u) {
    i6 = ((pi[((t) & 15u)]) & (255u));
    i7 = min(atomicLoad(&index_u32[((6815744u + i6))]), 4096u);
    i8 = 0u;
    for (k = 0u; k < min((lit_u32(4096u)), 16777216u); k = k + 256u) {
    if (((k + lid.x) < i7)) {
    i9 = atomicLoad(&index_u32[(((6841856u + (i6 * 4096u)) + (((k + lid.x)) & (4095u))))]);
    i10 = ((i9) & (((u32((lit_u32(32767u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))));
    i11 = min(i10, 25599u);
    b0 = (i10 < cfg.arrived_end);
    if ((b0 & (((i9) & (((u32((lit_u32(32768u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(0u))))) != 0u))) {
    i12 = atomicLoad(&index_u32[((6816256u + i11))]);
    for (k2 = 0u; k2 < min((lit_u32(8u)), 16777216u); k2 = k2 + 1u) {
    if ((pi[((k2) & 15u)] == i12)) {
    b0 = false;
    }
    }
    }
    if (b0) {
    i13 = (0u + (i11 * 256u));
    f9 = 0.0f;
    for (k2 = 0u; k2 < min((lit_u32(256u)), 16777216u); k2 = k2 + 1u) {
    f9 = (f9 + dot(unpack4x8snorm(u32(atomicLoad(&index_u32[((i13 + k2))]))), wg_va[((k2) & 255u)]));
    }
    f10 = (f9 * index_f32[((262144u + i11))]);
    if ((f10 > pa[3u])) {
    f11 = f10;
    i14 = i10;
    i15 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((f11 > pa[((k2) & 63u)])) {
    f12 = pa[((k2) & 63u)];
    i16 = pj[((k2) & 15u)];
    pa[((k2) & 63u)] = f11;
    pj[((k2) & 15u)] = i14;
    f11 = f12;
    i14 = i16;
    }
    }
    }
    }
    }
    }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    wg_c[((((lid.x * 4u) + k)) & 1023u)] = pa[((k) & 63u)];
    wg_uc[((((lid.x * 4u) + k)) & 1023u)] = pj[((k) & 15u)];
    }
    workgroupBarrier();
    for (k = 0u; k < 4u; k = k + 1u) {
    i17 = wg_uc[((((lid.x * 4u) + k)) & 1023u)];
    if ((i17 != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    f13 = wg_c[((((lid.x * 4u) + k)) & 1023u)];
    i18 = 0u;
    i19 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(1024u)), 16777216u); k2 = k2 + 1u) {
    if (((wg_uc[((k2) & 1023u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)))) & ((wg_c[((k2) & 1023u)] > f13) | ((wg_c[((k2) & 1023u)] == f13) & (k2 < ((lid.x * 4u) + k)))))) {
    i18 = (i18 + 1u);
    }
    }
    if (((i18 < 64u) & (wid.x < cfg.n_queries))) {
    cands[(((wid.x * 64u) + i18))] = (i17);
    }
    }
    }
}
