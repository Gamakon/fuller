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
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0;
    var f6: f32 = 0.0; var f7: f32 = 0.0; var f8: f32 = 0.0; var f9: f32 = 0.0;
    var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0;
    var f14: f32 = 0.0; var f15: f32 = 0.0; var f16r: f32 = 0.0; var f17: f32 = 0.0;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    for (k = 0u; k < 64u; k = k + 1u) {
        pa[k & 63u] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            f0 = 0.0f;
            f1 = 0.0f;
            if (((t + (((lidx + (256u * k))) >> 6u)) < cfg.m)) {
                if ((((wid.x * 64u) + (((lidx + (256u * k))) & 63u)) < cfg.d)) {
                    f0 = phi[((((t + (((lidx + (256u * k))) >> 6u)) * cfg.d) + (wid.x * 64u)) + (((lidx + (256u * k))) & 63u))];
                }
                if (((((wid.y * 64u) + (((lidx + (256u * k))) & 63u)) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + (((lidx + (256u * k))) & 63u)) < cfg.d))) {
                    f1 = phi[(((((t + (((lidx + (256u * k))) >> 6u)) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + (((lidx + (256u * k))) & 63u))];
                }
            }
            wg_a[(lidx + (256u * k)) & 2047u] = f0;
            wg_b[(lidx + (256u * k)) & 2047u] = f1;
        }
        workgroupBarrier();
        f2 = 0.0f; f3 = 0.0f; f4 = 0.0f; f5 = 0.0f;
        f6 = 0.0f; f7 = 0.0f; f8 = 0.0f; f9 = 0.0f;
        f10 = 0.0f; f11 = 0.0f; f12 = 0.0f; f13 = 0.0f;
        f14 = 0.0f; f15 = 0.0f; f16r = 0.0f; f17 = 0.0f;
        for (k = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let b0 = wg_b[(o + lid.y) & 2047u];
            let b1 = wg_b[(o + lid.y + 16u) & 2047u];
            let b2 = wg_b[(o + lid.y + 32u) & 2047u];
            let b3 = wg_b[(o + lid.y + 48u) & 2047u];
            let a0 = wg_a[(o + lid.x) & 2047u];
            let a1 = wg_a[(o + lid.x + 16u) & 2047u];
            let a2 = wg_a[(o + lid.x + 32u) & 2047u];
            let a3 = wg_a[(o + lid.x + 48u) & 2047u];
            f2 = f2 + b0 * a0;
            f3 = f3 + b0 * a1;
            f4 = f4 + b0 * a2;
            f5 = f5 + b0 * a3;
            f6 = f6 + b1 * a0;
            f7 = f7 + b1 * a1;
            f8 = f8 + b1 * a2;
            f9 = f9 + b1 * a3;
            f10 = f10 + b2 * a0;
            f11 = f11 + b2 * a1;
            f12 = f12 + b2 * a2;
            f13 = f13 + b2 * a3;
            f14 = f14 + b3 * a0;
            f15 = f15 + b3 * a1;
            f16r = f16r + b3 * a2;
            f17 = f17 + b3 * a3;
        }
        let q = ((t >> 5u) & 3u) * 16u;
        pa[q & 63u] = pa[q & 63u] + f2;
        pa[(q + 1u) & 63u] = pa[(q + 1u) & 63u] + f3;
        pa[(q + 2u) & 63u] = pa[(q + 2u) & 63u] + f4;
        pa[(q + 3u) & 63u] = pa[(q + 3u) & 63u] + f5;
        pa[(q + 4u) & 63u] = pa[(q + 4u) & 63u] + f6;
        pa[(q + 5u) & 63u] = pa[(q + 5u) & 63u] + f7;
        pa[(q + 6u) & 63u] = pa[(q + 6u) & 63u] + f8;
        pa[(q + 7u) & 63u] = pa[(q + 7u) & 63u] + f9;
        pa[(q + 8u) & 63u] = pa[(q + 8u) & 63u] + f10;
        pa[(q + 9u) & 63u] = pa[(q + 9u) & 63u] + f11;
        pa[(q + 10u) & 63u] = pa[(q + 10u) & 63u] + f12;
        pa[(q + 11u) & 63u] = pa[(q + 11u) & 63u] + f13;
        pa[(q + 12u) & 63u] = pa[(q + 12u) & 63u] + f14;
        pa[(q + 13u) & 63u] = pa[(q + 13u) & 63u] + f15;
        pa[(q + 14u) & 63u] = pa[(q + 14u) & 63u] + f16r;
        pa[(q + 15u) & 63u] = pa[(q + 15u) & 63u] + f17;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        if ((((((wid.y * 64u) + lid.y) + (16u * k)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + lid.y) + (16u * k))) < cfg.d))) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((((wid.x * 64u) + lid.x) + (16u * k2)) < cfg.d)) {
                    let idx = (((wid.y * 64u + lid.y) + k * 16u)) * cfg.d + ((wid.x * 64u + lid.x) + k2 * 16u);
                    let s = (k * 4u) + k2;
                    h[idx] = h[idx] + ((pa[s & 63u] + pa[(s + 16u) & 63u]) + (pa[(s + 32u) & 63u] + pa[(s + 48u) & 63u]));
                }
            }
        }
    }
}
