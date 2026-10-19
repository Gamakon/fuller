// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 64>;
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0;
    var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var g0: f32 = 0.0; var g1: f32 = 0.0;
    var t: u32 = 0u; var k: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let tid = lid.y * 16u + lid.x;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let idx = tid + 256u * k;
            let it = t + (idx >> 6u);
            let cc = idx & 63u;
            g0 = 0.0;
            g1 = 0.0;
            if (it < cfg.m) {
                if (colBase + cc < cfg.d) {
                    g0 = phi[it * cfg.d + colBase + cc];
                }
                if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < cfg.d) {
                    g1 = phi[it * cfg.d + cfg.r_off + rowBase + cc];
                }
            }
            wg_a[idx] = g0;
            wg_b[idx] = g1;
        }
        workgroupBarrier();
        f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0;
        f4 = 0.0; f5 = 0.0; f6 = 0.0; f7 = 0.0;
        f8 = 0.0; f9 = 0.0; f10 = 0.0; f11 = 0.0;
        f12 = 0.0; f13 = 0.0; f14 = 0.0; f15 = 0.0;
        for (k = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_a[k * 64u + lid.x];
            let a1 = wg_a[k * 64u + lid.x + 16u];
            let a2 = wg_a[k * 64u + lid.x + 32u];
            let a3 = wg_a[k * 64u + lid.x + 48u];
            let b0 = wg_b[k * 64u + lid.y];
            let b1 = wg_b[k * 64u + lid.y + 16u];
            let b2 = wg_b[k * 64u + lid.y + 32u];
            let b3 = wg_b[k * 64u + lid.y + 48u];
            f0 = f0 + b0 * a0; f1 = f1 + b0 * a1; f2 = f2 + b0 * a2; f3 = f3 + b0 * a3;
            f4 = f4 + b1 * a0; f5 = f5 + b1 * a1; f6 = f6 + b1 * a2; f7 = f7 + b1 * a3;
            f8 = f8 + b2 * a0; f9 = f9 + b2 * a1; f10 = f10 + b2 * a2; f11 = f11 + b2 * a3;
            f12 = f12 + b3 * a0; f13 = f13 + b3 * a1; f14 = f14 + b3 * a2; f15 = f15 + b3 * a3;
        }
        pa[0] = pa[0] + f0; pa[1] = pa[1] + f1; pa[2] = pa[2] + f2; pa[3] = pa[3] + f3;
        pa[4] = pa[4] + f4; pa[5] = pa[5] + f5; pa[6] = pa[6] + f6; pa[7] = pa[7] + f7;
        pa[8] = pa[8] + f8; pa[9] = pa[9] + f9; pa[10] = pa[10] + f10; pa[11] = pa[11] + f11;
        pa[12] = pa[12] + f12; pa[13] = pa[13] + f13; pa[14] = pa[14] + f14; pa[15] = pa[15] + f15;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = rowBase + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = colBase + lid.x + 16u * k2;
                if (c < cfg.d) {
                    let ix = lr * cfg.d + c;
                    h[ix] = h[ix] + pa[k * 4u + k2];
                }
            }
        }
    }
}
