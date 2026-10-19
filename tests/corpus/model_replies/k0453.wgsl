// grid 32 4 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(32, 4, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 32>;
    var pb: array<f32, 32>;
    var rb0: array<f32, 8>;
    var rb1: array<f32, 8>;
    var cb0: array<f32, 4>;
    var cb1: array<f32, 4>;
    var pm: array<bool, 8>;
    var pn: array<bool, 4>;
    let d = cfg.d;
    let row0 = wid.y * 32u + lid.y * 8u;
    let col0 = wid.x * 128u + lid.x;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pm[k] = ((row0 + k) < cfg.r_rows) && ((cfg.r_off + row0 + k) < d);
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        pn[k] = (col0 + k * 32u) < d;
    }
    let half = cfg.m / 2u;
    for (var t: u32 = 0u; t < half; t = t + 1u) {
        let base0 = (t * 2u) * d;
        let base1 = base0 + d;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            var a0: f32 = 0.0;
            var a1: f32 = 0.0;
            if (pm[k]) {
                a0 = phi[base0 + cfg.r_off + row0 + k];
                a1 = phi[base1 + cfg.r_off + row0 + k];
            }
            rb0[k] = a0;
            rb1[k] = a1;
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            var c0: f32 = 0.0;
            var c1: f32 = 0.0;
            if (pn[k]) {
                c0 = phi[base0 + col0 + k * 32u];
                c1 = phi[base1 + col0 + k * 32u];
            }
            cb0[k] = c0;
            cb1[k] = c1;
        }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                pa[k * 4u + k2] = pa[k * 4u + k2] + rb0[k] * cb0[k2];
                pb[k * 4u + k2] = pb[k * 4u + k2] + rb1[k] * cb1[k2];
            }
        }
    }
    if ((cfg.m & 1u) == 1u) {
        let base0 = (cfg.m - 1u) * d;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            var a0: f32 = 0.0;
            if (pm[k]) {
                a0 = phi[base0 + cfg.r_off + row0 + k];
            }
            rb0[k] = a0;
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            var c0: f32 = 0.0;
            if (pn[k]) {
                c0 = phi[base0 + col0 + k * 32u];
            }
            cb0[k] = c0;
        }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                pa[k * 4u + k2] = pa[k * 4u + k2] + rb0[k] * cb0[k2];
            }
        }
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
            if (pm[k] && pn[k2]) {
                let idx = (row0 + k) * d + col0 + k2 * 32u;
                h[idx] = h[idx] + (pa[k * 4u + k2] + pb[k * 4u + k2]);
            }
        }
    }
}
