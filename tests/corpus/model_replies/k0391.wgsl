// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        pa[q] = 0.0;
    }
    let row0 = gid.y * 4u;
    let col0 = gid.x * 4u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let lr = row0 + i;
            let gr = cfg.r_off + lr;
            var av: f32 = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                av = phi[base + gr];
            }
            pb[i] = av;
            let c = col0 + i;
            var bv: f32 = 0.0;
            if (c < cfg.d) {
                bv = phi[base + c];
            }
            pc[i] = bv;
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + pb[i] * pc[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = row0 + i;
            let c = col0 + j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
