// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    if (((gid.y * 1u)) < cfg.r_rows && ((gid.x * 1u)) < cfg.d && (cfg.r_off + ((gid.y * 1u))) < cfg.d) {
        let gr = cfg.r_off + gid.y;
        let c = gid.x;
        for (var tt: u32 = 0u; tt < cfg.m; tt = tt + 1u) {
            let base = tt * cfg.d;
            f0 = f0 + phi[base + gr] * phi[base + c];
        }
        h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] = h[(((gid.y * 1u)) * cfg.d + ((gid.x * 1u)))] + f0;
    }
}
