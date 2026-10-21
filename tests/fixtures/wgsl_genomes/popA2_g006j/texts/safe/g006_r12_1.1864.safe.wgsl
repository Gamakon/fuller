// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var t: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 1u) {
    f0 = (f0 + (phi[((t * cfg.d) + min((cfg.r_off + gid.y), select((cfg.d) - (1u), 0u, (1u) > (cfg.d))))] * phi[((t * cfg.d) + min(gid.x, select((cfg.d) - (1u), 0u, (1u) > (cfg.d))))]));
    }
    if (((((gid.y * 1u) < cfg.r_rows) & ((gid.x * 1u) < cfg.d)) & ((cfg.r_off + (gid.y * 1u)) < cfg.d))) {
    if ((((gid.y * 1u)) < cfg.r_rows) & (((gid.x * 1u)) < cfg.d) & ((cfg.r_off + ((gid.y * 1u))) < cfg.d)) { h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] = h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] + f0; }
    }
}
