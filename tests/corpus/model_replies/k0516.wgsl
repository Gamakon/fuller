// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    for (k = 0u; k < 64u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let e = lidx + 256u * k;
            let row = e >> 6u;
            let col = e & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (t + row < cfg.m) {
                if (wid.x * 64u + col < cfg.d) {
                    f0 = phi[(t + row) * cfg.d + wid.x * 64u + col];
                }
                if (wid.y * 64u + col < cfg.r_rows && cfg.r_off + wid.y * 64u + col < cfg.d) {
                    f1 = phi[(t + row) * cfg.d + cfg.r_off + wid.y * 64u + col];
                }
            }
            wg_a[e] = f0;
            wg_b[e] = f1;
        }
        workgroupBarrier();
        var c0: f32 = 0.0; var c1: f32 = 0.0; var c2: f32 = 0.0; var c3: f32 = 0.0;
        var c4: f32 = 0.0; var c5: f32 = 0.0; var c6: f32 = 0.0; var c7: f32 = 0.0;
        var c8: f32 = 0.0; var c9: f32 = 0.0; var c10: f32 = 0.0; var c11: f32 = 0.0;
        var c12: f32 = 0.0; var c13: f32 = 0.0; var c14: f32 = 0.0; var c15: f32 = 0.0;
        for (k = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let b0 = wg_b[o + lid.y];
            let b1 = wg_b[o + lid.y + 16u];
            let b2 = wg_b[o + lid.y + 32u];
            let b3 = wg_b[o + lid.y + 48u];
            let a0 = wg_a[o + lid.x];
            let a1 = wg_a[o + lid.x + 16u];
            let a2 = wg_a[o + lid.x + 32u];
            let a3 = wg_a[o + lid.x + 48u];
            c0 = c0 + b0 * a0; c1 = c1 + b0 * a1; c2 = c2 + b0 * a2; c3 = c3 + b0 * a3;
            c4 = c4 + b1 * a0; c5 = c5 + b1 * a1; c6 = c6 + b1 * a2; c7 = c7 + b1 * a3;
            c8 = c8 + b2 * a0; c9 = c9 + b2 * a1; c10 = c10 + b2 * a2; c11 = c11 + b2 * a3;
            c12 = c12 + b3 * a0; c13 = c13 + b3 * a1; c14 = c14 + b3 * a2; c15 = c15 + b3 * a3;
        }
        let s = ((t >> 5u) & 3u) * 16u;
        pa[s] = pa[s] + c0; pa[s + 1u] = pa[s + 1u] + c1; pa[s + 2u] = pa[s + 2u] + c2; pa[s + 3u] = pa[s + 3u] + c3;
        pa[s + 4u] = pa[s + 4u] + c4; pa[s + 5u] = pa[s + 5u] + c5; pa[s + 6u] = pa[s + 6u] + c6; pa[s + 7u] = pa[s + 7u] + c7;
        pa[s + 8u] = pa[s + 8u] + c8; pa[s + 9u] = pa[s + 9u] + c9; pa[s + 10u] = pa[s + 10u] + c10; pa[s + 11u] = pa[s + 11u] + c11;
        pa[s + 12u] = pa[s + 12u] + c12; pa[s + 13u] = pa[s + 13u] + c13; pa[s + 14u] = pa[s + 14u] + c14; pa[s + 15u] = pa[s + 15u] + c15;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = wid.x * 64u + lid.x + 16u * k2;
                if (c < cfg.d) {
                    let q = k * 4u + k2;
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + ((pa[q] + pa[q + 16u]) + (pa[q + 32u] + pa[q + 48u]));
                }
            }
        }
    }
}
