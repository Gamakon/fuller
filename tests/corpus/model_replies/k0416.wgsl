// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var t: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    for (i = 0u; i < 32u; i = i + 1u) {
        pa[i] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (i = 0u; i < 8u; i = i + 1u) {
            let lr = gid.y * 8u + i;
            let gr = cfg.r_off + lr;
            pb[i] = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                pb[i] = phi[base + gr];
            }
        }
        for (j = 0u; j < 4u; j = j + 1u) {
            let c = gid.x * 4u + j;
            pc[j] = 0.0;
            if (c < cfg.d) {
                pc[j] = phi[base + c];
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
            let lr = gid.y * 8u + i;
            let c = gid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
