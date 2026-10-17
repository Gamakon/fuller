// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + lid.y) < cfg.m)) {
    if ((((wid.x * 16u) + lid.x) < cfg.d)) {
    f0 = phi[(((t + lid.y) * cfg.d) + ((wid.x * 16u) + lid.x))];
    }
    if (((((wid.y * 16u) + lid.x) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 16u) + lid.x)) < cfg.d))) {
    f1 = phi[(((t + lid.y) * cfg.d) + (cfg.r_off + ((wid.y * 16u) + lid.x)))];
    }
    }
    wg_a[((((lid.y * 16u) + lid.x)) & 255u)] = f0;
    wg_b[((((lid.y * 16u) + lid.x)) & 255u)] = f1;
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 1u) {
    f2 = (f2 + (wg_b[((((k * 16u) + lid.y)) & 255u)] * wg_a[((((k * 16u) + lid.x)) & 255u)]));
    }
    workgroupBarrier();
    }
    if (((gid.y * 1u)) < cfg.r_rows && ((gid.x * 1u)) < cfg.d && (cfg.r_off + ((gid.y * 1u))) < cfg.d) { h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] = h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] + f2; }
}
