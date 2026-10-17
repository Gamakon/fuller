// grid 13 39 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
fn lit_u32(x: u32) -> u32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
fn at_vec4b(v: vec4<bool>, i: u32) -> bool { var t = v; return t[i & 3u]; }
@compute @workgroup_size(13, 39, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(subgroup_size) sg_size: u32, @builtin(num_subgroups) sg_count: u32) {
    let sg_lane: u32 = subgroupExclusiveAdd(1u);
    var t: u32 = 0u;
    var k: u32 = 0u;
    var pn: array<bool, 16>;
    var pvq: array<vec4<f32>, 64>;
    for (; t < cfg.d; t = t + 1u) {
    storageBarrier();
    workgroupBarrier();
    subgroupBarrier();
    }
    if ((nwg.x) <= (wid.z)) {
    storageBarrier();
    if ((nwg.y) != (lit_u32(0u))) {
    if ((cfg.r_rows <= arrayLength(&phi))) {
    }
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
    pvq[0u] = vec4<f32>(-2.1661954f);
    for (k = 0u; k < 2u; k = k + 1u) {
    }
    }
    } else {
    storageBarrier();
    }
    for (t = 0u; t < min((((cfg.d / max((nwg.y % max(cfg.r_rows, 1u)), 1u)) / max(((min(nwg.y, max(sg_count, 1u) - 1u)) & (sg_size)), 1u))), 16777216u); t = t + 1u) {
    if ((5u <= cfg.r_off)) { continue; }
    if (at_vec4b(vec4<bool>((cfg.r_off >= 5u)), 0u)) { continue; }
    if (pn[((sg_lane) & 15u)]) { continue; }
    }
    if (((gid.y * 8u)) < cfg.r_rows && ((wid.x * 104u + lid.x)) < cfg.d && (cfg.r_off + ((gid.y * 8u))) < cfg.d) { h[(((gid.y * 8u)) * cfg.d + ((wid.x * 104u + lid.x)))] = h[(((gid.y * 8u)) * cfg.d + ((wid.x * 104u + lid.x)))] + phi[(min(clamp(vec4<u32>(lit_u32(0u)), vec4<u32>(lit_u32(0u)), vec4<u32>(lit_u32(0u))), vec4<u32>(vec4<i32>(lit_i32(0i)))).z)]; }
}

