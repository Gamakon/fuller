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
    var k: u32 = 0u;
    var t: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k & 63u] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 6u);
            let col = idx & 63u;
            var va: f32 = 0.0f;
            var vb: f32 = 0.0f;
            if (item < cfg.m) {
                if (wid.x * 64u + col < cfg.d) {
                    va = phi[item * cfg.d + wid.x * 64u + col];
                }
                if (wid.y * 64u + col < cfg.r_rows && cfg.r_off + wid.y * 64u + col < cfg.d) {
                    vb = phi[item * cfg.d + cfg.r_off + wid.y * 64u + col];
                }
            }
            wg_a[idx & 2047u] = va;
            wg_b[idx & 2047u] = vb;
        }
        workgroupBarrier();
        var f0: f32 = 0.0f; var f1: f32 = 0.0f; var f2: f32 = 0.0f; var f3: f32 = 0.0f;
        var f4: f32 = 0.0f; var f5: f32 = 0.0f; var f6: f32 = 0.0f; var f7: f32 = 0.0f;
        var f8: f32 = 0.0f; var f9: f32 = 0.0f; var f10: f32 = 0.0f; var f11: f32 = 0.0f;
        var f12: f32 = 0.0f; var f13: f32 = 0.0f; var f14: f32 = 0.0f; var f15: f32 = 0.0f;
        for (k = 0u; k < 32u; k = k + 1u) {
            let ba = k * 64u + lid.y;
            let aa = k * 64u + lid.x;
            let b0 = wg_b[ba & 2047u];
            let b1 = wg_b[(ba + 16u) & 2047u];
            let b2 = wg_b[(ba + 32u) & 2047u];
            let b3 = wg_b[(ba + 48u) & 2047u];
            let a0 = wg_a[aa & 2047u];
            let a1 = wg_a[(aa + 16u) & 2047u];
            let a2 = wg_a[(aa + 32u) & 2047u];
            let a3 = wg_a[(aa + 48u) & 2047u];
            f0 = f0 + b0 * a0; f1 = f1 + b0 * a1; f2 = f2 + b0 * a2; f3 = f3 + b0 * a3;
            f4 = f4 + b1 * a0; f5 = f5 + b1 * a1; f6 = f6 + b1 * a2; f7 = f7 + b1 * a3;
            f8 = f8 + b2 * a0; f9 = f9 + b2 * a1; f10 = f10 + b2 * a2; f11 = f11 + b2 * a3;
            f12 = f12 + b3 * a0; f13 = f13 + b3 * a1; f14 = f14 + b3 * a2; f15 = f15 + b3 * a3;
        }
        pa[0u] = pa[0u] + f0; pa[1u] = pa[1u] + f1; pa[2u] = pa[2u] + f2; pa[3u] = pa[3u] + f3;
        pa[4u] = pa[4u] + f4; pa[5u] = pa[5u] + f5; pa[6u] = pa[6u] + f6; pa[7u] = pa[7u] + f7;
        pa[8u] = pa[8u] + f8; pa[9u] = pa[9u] + f9; pa[10u] = pa[10u] + f10; pa[11u] = pa[11u] + f11;
        pa[12u] = pa[12u] + f12; pa[13u] = pa[13u] + f13; pa[14u] = pa[14u] + f14; pa[15u] = pa[15u] + f15;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let row = wid.y * 64u + lid.y + 16u * k;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = wid.x * 64u + lid.x + 16u * k2;
                if (col < cfg.d) {
                    h[row * cfg.d + col] = h[row * cfg.d + col] + pa[(k * 4u + k2) & 63u];
                }
            }
        }
    }
}
