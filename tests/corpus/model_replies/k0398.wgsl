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
    for (var q: u32 = 0u; q < 32u; q = q + 1u) {
        pa[q] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let lr = gid.y * 8u + k;
            let gr = cfg.r_off + lr;
            var av: f32 = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                av = phi[base + gr];
            }
            pb[k] = av;
        }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = gid.x * 4u + j;
            var cv: f32 = 0.0;
            if (c < cfg.d) {
                cv = phi[base + c];
            }
            pc[j] = cv;
        }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[k * 4u + j] = pa[k * 4u + j] + pb[k] * pc[j];
            }
        }
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        let gr = cfg.r_off + lr;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = gid.x * 4u + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[k * 4u + j];
            }
        }
    }
}
