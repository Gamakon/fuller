// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var t: u32 = 0u;
    i0 = (cfg.r_off + gid.y);
    i1 = gid.x;
    for (t = 0u; t < min((((cfg.m) - ((cfg.m) % max(4u, 1u)))), 16777216u); t = t + 4u) {
    f0 = fma(phi[((t * cfg.d) + (cfg.r_off + gid.y))], phi[((t * cfg.d) + gid.x)], f0);
    f1 = fma(phi[(((t * cfg.d) + cfg.d) + (cfg.r_off + gid.y))], phi[(((t * cfg.d) + cfg.d) + gid.x)], f1);
    f2 = fma(phi[((((t * cfg.d) + cfg.d) + cfg.d) + (cfg.r_off + gid.y))], phi[((((t * cfg.d) + cfg.d) + cfg.d) + gid.x)], f2);
    f3 = fma(phi[(((((t * cfg.d) + cfg.d) + cfg.d) + cfg.d) + (cfg.r_off + gid.y))], phi[(((((t * cfg.d) + cfg.d) + cfg.d) + cfg.d) + gid.x)], f3);
    }
    for (; t < cfg.m; t = t + 1u) {
    f0 = fma(phi[((t * cfg.d) + (cfg.r_off + gid.y))], phi[((t * cfg.d) + gid.x)], f0);
    }
    if ((!(((gid.y >= cfg.r_rows) || (gid.x >= cfg.d)) || ((cfg.r_off + gid.y) >= cfg.d)))) {
    if (((gid.y * 1u)) < cfg.r_rows && ((gid.x * 1u)) < cfg.d && (cfg.r_off + ((gid.y * 1u))) < cfg.d) { h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] = h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] + ((f0 + f1) + (f2 + f3)); }
    }
}
