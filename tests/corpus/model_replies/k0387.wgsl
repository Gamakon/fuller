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
        for (k = 0u; k < 4u; k = k + 1u) {
            let c = gid.x * 4u + k;
            var cv: f32 = 0.0;
            if (c < cfg.d) {
                cv = phi[base + c];
            }
            pb[k] = cv;
        }
        for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
            let lr = gid.y * 8u + k2;
            let gr = cfg.r_off + lr;
            var rv: f32 = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                rv = phi[base + gr];
            }
            pc[k2] = rv;
        }
        for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
            for (k = 0u; k < 4u; k = k + 1u) {
                pa[k2 * 4u + k] = pa[k2 * 4u + k] + pc[k2] * pb[k];
            }
        }
    }
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
        let lr = gid.y * 8u + k2;
        for (k = 0u; k < 4u; k = k + 1u) {
            let c = gid.x * 4u + k;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[k2 * 4u + k];
            }
        }
    }
}
