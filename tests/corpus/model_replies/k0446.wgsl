// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = cfg.d;
    let rb = gid.y * 8u;
    let cb = gid.x * 4u;
    var acc: array<f32, 32>;
    var a0: array<f32, 8>;
    var a1: array<f32, 8>;
    var b0: array<f32, 4>;
    var b1: array<f32, 4>;
    for (var i = 0u; i < 32u; i = i + 1u) { acc[i] = 0.0; }
    let half = cfg.m / 2u;
    for (var t2 = 0u; t2 < half; t2 = t2 + 1u) {
        let o0 = (t2 * 2u) * d;
        let o1 = o0 + d;
        for (var k = 0u; k < 8u; k = k + 1u) {
            var x0 = 0.0; var x1 = 0.0;
            if (rb + k < cfg.r_rows && cfg.r_off + rb + k < d) {
                x0 = phi[o0 + cfg.r_off + rb + k];
                x1 = phi[o1 + cfg.r_off + rb + k];
            }
            a0[k] = x0; a1[k] = x1;
        }
        for (var k = 0u; k < 4u; k = k + 1u) {
            var y0 = 0.0; var y1 = 0.0;
            if (cb + k < d) {
                y0 = phi[o0 + cb + k];
                y1 = phi[o1 + cb + k];
            }
            b0[k] = y0; b1[k] = y1;
        }
        for (var k = 0u; k < 8u; k = k + 1u) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                acc[k * 4u + j] = acc[k * 4u + j] + (a0[k] * b0[j] + a1[k] * b1[j]);
            }
        }
    }
    if ((cfg.m & 1u) == 1u) {
        let o0 = (cfg.m - 1u) * d;
        for (var k = 0u; k < 8u; k = k + 1u) {
            var x0 = 0.0;
            if (rb + k < cfg.r_rows && cfg.r_off + rb + k < d) {
                x0 = phi[o0 + cfg.r_off + rb + k];
            }
            a0[k] = x0;
        }
        for (var k = 0u; k < 4u; k = k + 1u) {
            var y0 = 0.0;
            if (cb + k < d) { y0 = phi[o0 + cb + k]; }
            b0[k] = y0;
        }
        for (var k = 0u; k < 8u; k = k + 1u) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                acc[k * 4u + j] = acc[k * 4u + j] + a0[k] * b0[j];
            }
        }
    }
    for (var k = 0u; k < 8u; k = k + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            if (rb + k < cfg.r_rows && cfg.r_off + rb + k < d && cb + j < d) {
                let idx = (rb + k) * d + cb + j;
                h[idx] = h[idx] + acc[k * 4u + j];
            }
        }
    }
}
