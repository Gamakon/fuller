// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (var z: u32 = 0u; z < 64u; z = z + 1u) {
        pa[z] = 0.0;
        pb[z] = 0.0;
    }
    let row0 = gid.y * 8u;
    let col0 = gid.x * 4u;
    let half_m = cfg.m / 2u;
    for (var q: u32 = 0u; q < half_m; q = q + 1u) {
        let base0 = (2u * q) * cfg.d;
        let base1 = (2u * q + 1u) * cfg.d;
        var ra0: array<f32, 8>;
        var ra1: array<f32, 8>;
        var ca0: array<f32, 4>;
        var ca1: array<f32, 4>;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let lr = row0 + i;
            let gr = cfg.r_off + lr;
            var x0: f32 = 0.0;
            var x1: f32 = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                x0 = phi[base0 + gr];
                x1 = phi[base1 + gr];
            }
            ra0[i] = x0;
            ra1[i] = x1;
        }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + j;
            var y0: f32 = 0.0;
            var y1: f32 = 0.0;
            if (c < cfg.d) {
                y0 = phi[base0 + c];
                y1 = phi[base1 + c];
            }
            ca0[j] = y0;
            ca1[j] = y1;
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + ra0[i] * ca0[j];
                pb[i * 4u + j] = pb[i * 4u + j] + ra1[i] * ca1[j];
            }
        }
    }
    if ((cfg.m & 1u) == 1u) {
        let baseT = (cfg.m - 1u) * cfg.d;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let lr = row0 + i;
            let gr = cfg.r_off + lr;
            var x: f32 = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                x = phi[baseT + gr];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + j;
                var y: f32 = 0.0;
                if (c < cfg.d) {
                    y = phi[baseT + c];
                }
                pa[i * 4u + j] = pa[i * 4u + j] + x * y;
            }
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + i;
        let gr = cfg.r_off + lr;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + (pa[i * 4u + j] + pb[i * 4u + j]);
            }
        }
    }
}
