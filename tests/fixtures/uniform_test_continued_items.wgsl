// grid 54 18 8 8 rows-x
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_d: array<f32, 2048>;
var<workgroup> wg_ue: array<u32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
fn at_vec4b(v: vec4<bool>, i: u32) -> bool { var t = v; return t[i & 3u]; }
@compute @workgroup_size(54, 18, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(subgroup_size) sg_size: u32, @builtin(num_subgroups) sg_count: u32) {
    let sg_index: u32 = lidx / sg_size;
    var f41: f32 = 0.0;
    var v20: vec4<f32> = vec4<f32>(0.0);
    var v29: vec4<f32> = vec4<f32>(0.0);
    var i40: u32 = 0u;
    var b2: bool = false;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pm: array<bool, 16>;
    var pvr: array<vec4<f32>, 64>;
    if ((cfg.r_off) < ((((lit_u32(0u)) - ((lit_u32(0u)) % max(wid.x, 1u))) + min(cfg.r_off, max(min(nwg.z, 1u), 1u) - 1u)))) {
    if ((sg_count) <= (max(clamp(nwg.z, cfg.r_rows, nwg.y), sg_size))) {
    storageBarrier();
    for (t = 0u; t + 8u <= cfg.r_rows; t = t + 8u) {
        t = t + 1u;
        t = t + 1u;
        t = t + 1u;
        t = t + 1u;
        t = t + 1u;
        t = t + 1u;
        t = t + 1u;
        t = t - 7u;
    }
    for (; t < cfg.r_rows; t = t + 1u) {
    }
    if ((cfg.d) < (nwg.y)) {
    }
    for (; t < 1u; t = t + 1u) {
    }
    } else {
    subgroupBarrier();
    if ((sg_size) == (sg_size)) {
    }
    workgroupBarrier();
    workgroupBarrier();
    if (lidx < 8u) { if (wg_a[((lidx + 8u) & 2047u)] > wg_a[((lidx) & 2047u)]) { wg_a[((lidx) & 2047u)] = wg_a[((lidx + 8u) & 2047u)]; wg_ue[((lidx) & 2047u)] = wg_ue[((lidx + 8u) & 2047u)]; } }
    workgroupBarrier();
    if (lidx < 4u) { if (wg_a[((lidx + 4u) & 2047u)] > wg_a[((lidx) & 2047u)]) { wg_a[((lidx) & 2047u)] = wg_a[((lidx + 4u) & 2047u)]; wg_ue[((lidx) & 2047u)] = wg_ue[((lidx + 4u) & 2047u)]; } }
    workgroupBarrier();
    if (lidx < 2u) { if (wg_a[((lidx + 2u) & 2047u)] > wg_a[((lidx) & 2047u)]) { wg_a[((lidx) & 2047u)] = wg_a[((lidx + 2u) & 2047u)]; wg_ue[((lidx) & 2047u)] = wg_ue[((lidx + 2u) & 2047u)]; } }
    workgroupBarrier();
    if (lidx < 1u) { if (wg_a[((lidx + 1u) & 2047u)] > wg_a[((lidx) & 2047u)]) { wg_a[((lidx) & 2047u)] = wg_a[((lidx + 1u) & 2047u)]; wg_ue[((lidx) & 2047u)] = wg_ue[((lidx + 1u) & 2047u)]; } }
    workgroupBarrier();
    }
    subgroupBarrier();
    if ((nwg.y) <= (clamp(cfg.r_rows, cfg.r_rows, clamp(cfg.r_rows, cfg.r_off, sg_size)))) {
    if ((cfg.d) < (nwg.z)) {
    }
    }
    } else {
    if ((cfg.r_off) == (min(nwg.x, max(lit_u32(0u), 1u) - 1u))) {
    workgroupBarrier();
    if (lidx < 1u) { wg_va[((lidx) & 511u)] = (wg_va[((lidx) & 511u)] + wg_va[((lidx + 1u) & 511u)]); }
    workgroupBarrier();
    for (t = 0u; t < cfg.d; t = t + 1u) {
    }
    for (t = 0u; t < cfg.d; t = t + 1u) {
    }
    storageBarrier();
    } else {
    workgroupBarrier();
    if (lidx < 16u) { wg_a[((lidx) & 2047u)] = max(wg_a[((lidx) & 2047u)], wg_a[((lidx + 16u) & 2047u)]); }
    workgroupBarrier();
    if (lidx < 8u) { wg_a[((lidx) & 2047u)] = max(wg_a[((lidx) & 2047u)], wg_a[((lidx + 8u) & 2047u)]); }
    workgroupBarrier();
    if (lidx < 4u) { wg_a[((lidx) & 2047u)] = max(wg_a[((lidx) & 2047u)], wg_a[((lidx + 4u) & 2047u)]); }
    workgroupBarrier();
    if (lidx < 2u) { wg_a[((lidx) & 2047u)] = max(wg_a[((lidx) & 2047u)], wg_a[((lidx + 2u) & 2047u)]); }
    workgroupBarrier();
    if (lidx < 1u) { wg_a[((lidx) & 2047u)] = max(wg_a[((lidx) & 2047u)], wg_a[((lidx + 1u) & 2047u)]); }
    workgroupBarrier();
    workgroupBarrier();
    if (lidx < 2u) { wg_va[((lidx) & 511u)] = (wg_va[((lidx) & 511u)] + wg_va[((lidx + 2u) & 511u)]); }
    workgroupBarrier();
    if (lidx < 1u) { wg_va[((lidx) & 511u)] = (wg_va[((lidx) & 511u)] + wg_va[((lidx + 1u) & 511u)]); }
    workgroupBarrier();
    for (t = 0u; t < 1u; t = t + 1u) {
    }
    for (t = 0u; t + 2u <= 1u; t = t + 2u) {
        t = t + 1u;
        t = t - 1u;
    }
    for (; t < 1u; t = t + 1u) {
    }
    }
    }
    storageBarrier();
    f41 = workgroupUniformLoad(&wg_d[75u]);
    i40 = ((lit_u32(0u)) - ((lit_u32(0u)) % max((vec2<u32>(lit_u32(0u)).y / max(vec4<u32>(vec4<u32>(lit_u32(0u))).w, 1u)), 1u)));
    pvr[15u] = vec4<f32>(-2.0398269f);
    for (k = 0u; k < min((min(lit_u32(0u), max(cfg.d, 1u) - 1u)), 16777216u); k = k + 1u) {
    if (b2) { continue; }
    if (pm[((sg_index) & 15u)]) { continue; }
    v29 = vec4<f32>(-0.25546673f);
    if (at_vec4b((v20 != v20), 0u)) {
    } else {
    }
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    if (((gid.x * 8u)) < cfg.r_rows && ((gid.y * 8u)) < cfg.d && (cfg.r_off + ((gid.x * 8u))) < cfg.d) { h[(((gid.x * 8u)) * cfg.d + ((gid.y * 8u)))] = h[(((gid.x * 8u)) * cfg.d + ((gid.y * 8u)))] + phi[(lit_u32(8u))]; }
    }
    }
}

