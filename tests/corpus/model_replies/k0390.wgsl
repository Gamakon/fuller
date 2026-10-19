// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var k: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    var t: u32 = 0u;
    let row0 = gid.y * 8u;
    let col0 = gid.x * 4u;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (i = 0u; i < 8u; i = i + 1u) {
            pb[i] = 0.0;
            if (row0 + i < cfg.r_rows && cfg.r_off + row0 + i < cfg.d) {
                pb[i] = phi[base + cfg.r_off + row0 + i];
            }
        }
        for (j = 0u; j < 4u; j = j + 1u) {
            pc[j] = 0.0;
            if (col0 + j < cfg.d) {
                pc[j] = phi[base + col0 + j];
            }
        }
        for (i = 0u; i < 8u; i = i + 1u) {
            for (j = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + pb[i] * pc[j];
            }
        }
    }
    for (i = 0u; i < 8u; i = i + 1u) {
        for (j = 0u; j < 4u; j = j + 1u) {
            if (row0 + i < cfg.r_rows && cfg.r_off + row0 + i < cfg.d && col0 + j < cfg.d) {
                let idx = (row0 + i) * cfg.d + col0 + j;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
