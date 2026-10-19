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
    var k2: u32 = 0u;
    var t: u32 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (k = 0u; k < 8u; k = k + 1u) {
            let lr = gid.y * 8u + k;
            let gr = cfg.r_off + lr;
            pb[k] = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                pb[k] = phi[base + gr];
            }
        }
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let c = gid.x * 4u + k2;
            pc[k2] = 0.0;
            if (c < cfg.d) {
                pc[k2] = phi[base + c];
            }
        }
        for (k = 0u; k < 8u; k = k + 1u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                pa[k * 4u + k2] = pa[k * 4u + k2] + pb[k] * pc[k2];
            }
        }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        let gr = cfg.r_off + lr;
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let c = gid.x * 4u + k2;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 4u + k2];
            }
        }
    }
}
