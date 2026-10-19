// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let rbase = gid.y * 8u;
    let cbase = gid.x * 4u;
    for (t = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (k = 0u; k < 8u; k = k + 1u) {
            var av: f32 = 0.0;
            if ((rbase + k) < cfg.r_rows && (cfg.r_off + rbase + k) < cfg.d) {
                av = phi[base + cfg.r_off + rbase + k];
            }
            pb[k] = av;
        }
        for (k = 0u; k < 4u; k = k + 1u) {
            var bv: f32 = 0.0;
            if ((cbase + k) < cfg.d) {
                bv = phi[base + cbase + k];
            }
            pc[k] = bv;
        }
        for (k = 0u; k < 8u; k = k + 1u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                pa[k * 4u + k2] = pa[k * 4u + k2] + pb[k] * pc[k2];
            }
        }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            if ((rbase + k) < cfg.r_rows && (cbase + k2) < cfg.d && (cfg.r_off + rbase + k) < cfg.d) {
                let idx = (rbase + k) * cfg.d + cbase + k2;
                h[idx] = h[idx] + pa[k * 4u + k2];
            }
        }
    }
}
