// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var c0: f32 = 0.0; var c1: f32 = 0.0; var c2: f32 = 0.0; var c3: f32 = 0.0;
    var c4: f32 = 0.0; var c5: f32 = 0.0; var c6: f32 = 0.0; var c7: f32 = 0.0;
    var c8: f32 = 0.0; var c9: f32 = 0.0; var c10: f32 = 0.0; var c11: f32 = 0.0;
    var c12: f32 = 0.0; var c13: f32 = 0.0; var c14: f32 = 0.0; var c15: f32 = 0.0;
    for (k = 0u; k < 64u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let li = lidx + 256u * k;
            let item = t + (li >> 6u);
            let cc = li & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (item < cfg.m) {
                if (wid.x * 64u + cc < cfg.d) {
                    f0 = phi[item * cfg.d + wid.x * 64u + cc];
                }
                if (wid.y * 64u + cc < cfg.r_rows && cfg.r_off + wid.y * 64u + cc < cfg.d) {
                    f1 = phi[item * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[li] = f0;
            wg_b[li] = f1;
        }
        workgroupBarrier();
        c0 = 0.0; c1 = 0.0; c2 = 0.0; c3 = 0.0; c4 = 0.0; c5 = 0.0; c6 = 0.0; c7 = 0.0;
        c8 = 0.0; c9 = 0.0; c10 = 0.0; c11 = 0.0; c12 = 0.0; c13 = 0.0; c14 = 0.0; c15 = 0.0;
        for (k = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let a0 = wg_a[o + lid.x];
            let a1 = wg_a[o + lid.x + 32u];
            let b0 = wg_b[o + lid.y];
            let b1 = wg_b[o + lid.y + 8u];
            let b2 = wg_b[o + lid.y + 16u];
            let b3 = wg_b[o + lid.y + 24u];
            let b4 = wg_b[o + lid.y + 32u];
            let b5 = wg_b[o + lid.y + 40u];
            let b6 = wg_b[o + lid.y + 48u];
            let b7 = wg_b[o + lid.y + 56u];
            c0 = c0 + b0 * a0; c1 = c1 + b0 * a1;
            c2 = c2 + b1 * a0; c3 = c3 + b1 * a1;
            c4 = c4 + b2 * a0; c5 = c5 + b2 * a1;
            c6 = c6 + b3 * a0; c7 = c7 + b3 * a1;
            c8 = c8 + b4 * a0; c9 = c9 + b4 * a1;
            c10 = c10 + b5 * a0; c11 = c11 + b5 * a1;
            c12 = c12 + b6 * a0; c13 = c13 + b6 * a1;
            c14 = c14 + b7 * a0; c15 = c15 + b7 * a1;
        }
        let s = ((t >> 5u) & 3u) * 16u;
        pa[s] = pa[s] + c0;
        pa[s + 1u] = pa[s + 1u] + c1;
        pa[s + 2u] = pa[s + 2u] + c2;
        pa[s + 3u] = pa[s + 3u] + c3;
        pa[s + 4u] = pa[s + 4u] + c4;
        pa[s + 5u] = pa[s + 5u] + c5;
        pa[s + 6u] = pa[s + 6u] + c6;
        pa[s + 7u] = pa[s + 7u] + c7;
        pa[s + 8u] = pa[s + 8u] + c8;
        pa[s + 9u] = pa[s + 9u] + c9;
        pa[s + 10u] = pa[s + 10u] + c10;
        pa[s + 11u] = pa[s + 11u] + c11;
        pa[s + 12u] = pa[s + 12u] + c12;
        pa[s + 13u] = pa[s + 13u] + c13;
        pa[s + 14u] = pa[s + 14u] + c14;
        pa[s + 15u] = pa[s + 15u] + c15;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let row = wid.y * 64u + lid.y + 8u * i;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let col = wid.x * 64u + lid.x + 32u * j;
                if (col < cfg.d) {
                    let n = i * 2u + j;
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + ((pa[n] + pa[n + 16u]) + (pa[n + 32u] + pa[n + 48u]));
                }
            }
        }
    }
}
