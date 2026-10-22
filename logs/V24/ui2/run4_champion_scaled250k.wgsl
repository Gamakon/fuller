// grid 128 1 1 1
// entry ingest grid 128 1 1 1
// entry search grid 64 1 1 1
struct Cfg { seed_slot: u32, stage: u32, arrived_begin: u32, arrived_end: u32, n_queries: u32, pass_index: u32, n_passes: u32, c_max: u32, };
@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> index_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<u32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 128>;
var<workgroup> wg_b: array<f32, 128>;
var<workgroup> wg_c: array<f32, 512>;
var<workgroup> wg_ua: array<u32, 512>;
var<workgroup> wg_va: array<vec4<f32>, 128>;
fn lit_u32(x: u32) -> u32 { return x; }
@compute @workgroup_size(128, 1, 1)
fn ingest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var f4: f32 = 0.0;
    var f5: f32 = 0.0;
    var i1: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    f0 = (f0 + (select(-1.0f, 1.0f, (((((((((((((4u * lid.x) + k) + (512u * k2)) * 1237u) + 311u)) & (2047u)) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))))) >> ((13u) & 31u))) & (1u)) == 0u)) * keys[(((((wid.x) & (65535u)) * 2048u) + (((((((4u * lid.x) + k) + (512u * k2)) * 1237u) + 311u)) & (2047u))))]));
    }
    pa[((k) & 63u)] = f0;
    f1 = max(f1, abs(f0));
    f2 = (f2 + (f0 * f0));
    }
    wg_a[((lid.x) & 127u)] = f1;
    wg_b[((lid.x) & 127u)] = f2;
    workgroupBarrier();
    for (k = 0u; k < 7u; k = k + 1u) {
    if ((lid.x < ((64u) >> ((k) & 31u)))) {
    wg_a[((lid.x) & 127u)] = max(wg_a[((lid.x) & 127u)], wg_a[(((lid.x + ((64u) >> ((k) & 31u)))) & 127u)]);
    wg_b[((lid.x) & 127u)] = (wg_b[((lid.x) & 127u)] + wg_b[(((lid.x + ((64u) >> ((k) & 31u)))) & 127u)]);
    }
    workgroupBarrier();
    }
    f3 = wg_a[0u];
    f3 = wg_a[0u];
    f4 = wg_b[0u];
    if ((f3 > 0.0f)) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i1 = ((i1) | (u32(((u32(u32(i32(clamp(round(((pa[((k) & 63u)] / f3) * 127.0f)), -127.0f, 127.0f)))) & 255u) << ((u32((8u * k))) & 31u)))));
    }
    }
    index_u32[((((((cfg.arrived_begin + wid.x)) & (262143u)) * 128u) + lid.x))] = (i1);
    if ((lid.x == 0u)) {
    if (((f3 > 0.0f) & (f4 > 0.0f))) {
    f5 = ((f3 / 127.0f) / sqrt(f4));
    }
    index_f32[((((cfg.arrived_begin + wid.x)) & (262143u)))] = (f5);
    }
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
    var v0: vec4<f32> = vec4<f32>(0.0);
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i4: u32 = 0u;
    var i5: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pa: array<f32, 64>;
    var pi: array<u32, 16>;
    for (k = 0u; k < 2u; k = k + 1u) {
    v0 = vec4<f32>(0.0f);
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    f0 = 0.0f;
    i1 = 0u;
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    f0 = (f0 + (select(-1.0f, 1.0f, (((((((((((((4u * (lid.x + (64u * k))) + k2) + (512u * k3)) * 1237u) + 311u)) & (2047u)) * ((u32((lit_u32(40503u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(31153u))))) >> ((13u) & 31u))) & (1u)) == 0u)) * queries[(((((wid.x) & (4095u)) * 2048u) + (((((((4u * (lid.x + (64u * k))) + k2) + (512u * k3)) * 1237u) + 311u)) & (2047u))))]));
    }
    v0[(k2) & 3u] = f0;
    }
    wg_va[(((lid.x + (64u * k))) & 127u)] = v0;
    }
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    pa[((k) & 63u)] = -3e38f;
    pi[((k) & 15u)] = ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u)));
    }
    for (t = 0u; t < min((cfg.arrived_end), 16777216u); t = t + 64u) {
    if (((t + lid.x) < cfg.arrived_end)) {
    f1 = 0.0f;
    for (k = 0u; k < min((lit_u32(128u)), 16777216u); k = k + 1u) {
    f1 = (f1 + dot(vec4<f32>(vec3<f32>(vec2<f32>(f32((i32(u32((u32(index_u32[((((((t + lid.x)) & (262143u)) * 128u) + k))]) << ((24u) & 31u)))) >> ((24u) & 31u))), f32((i32(u32((u32(index_u32[((((((t + lid.x)) & (262143u)) * 128u) + k))]) << ((16u) & 31u)))) >> ((24u) & 31u)))), f32((i32(u32((u32(index_u32[((((((t + lid.x)) & (262143u)) * 128u) + k))]) << ((8u) & 31u)))) >> ((24u) & 31u)))), f32((i32(index_u32[((((((t + lid.x)) & (262143u)) * 128u) + k))]) >> ((24u) & 31u)))), wg_va[((k) & 127u)]));
    }
    f2 = index_f32[((((t + lid.x)) & (262143u)))];
    f3 = -3e38f;
    if ((f2 > 0.0f)) {
    f3 = (f1 * f2);
    }
    if ((f3 > pa[7u])) {
    f4 = f3;
    i2 = (lid.x + t);
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((f4 > pa[((k) & 63u)])) {
    f5 = pa[((k) & 63u)];
    i4 = pi[((k) & 15u)];
    pa[((k) & 63u)] = f4;
    f4 = f5;
    pi[((k) & 15u)] = i2;
    i2 = i4;
    }
    }
    }
    }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    wg_c[((((lid.x * 8u) + k)) & 511u)] = pa[((k) & 63u)];
    wg_ua[((((lid.x * 8u) + k)) & 511u)] = pi[((k) & 15u)];
    }
    workgroupBarrier();
    if ((wid.x < cfg.n_queries)) {
    for (k = 0u; k < 8u; k = k + 1u) {
    f6 = wg_c[((((lid.x * 8u) + k)) & 511u)];
    if ((wg_ua[((((lid.x * 8u) + k)) & 511u)] != ((u32((lit_u32(65535u) << ((lit_u32(16u)) & 31u)))) | (lit_u32(65535u))))) {
    i5 = 0u;
    for (k2 = 0u; k2 < min((lit_u32(512u)), 16777216u); k2 = k2 + 1u) {
    if (((wg_c[((k2) & 511u)] > f6) | ((wg_c[((k2) & 511u)] == f6) & (k2 >= ((lid.x * 8u) + k))))) {
    i5 = (i5 + 1u);
    i5 = (i5 + 1u);
    }
    }
    if ((i5 < 64u)) {
    cands[(((wid.x * 64u) + i5))] = (wg_ua[((((lid.x * 8u) + k)) & 511u)]);
    }
    }
    }
    }
}
