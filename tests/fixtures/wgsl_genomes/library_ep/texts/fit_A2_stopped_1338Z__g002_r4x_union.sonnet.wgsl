// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 32>;
    for (var q: u32 = 0u; q < 32u; q = q + 1u) { pa[q] = 0.0; }
    let lidx = lid.y * 16u + lid.x;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let tt = t + 16u * j;
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                let idx = lidx + 256u * k;
                let item = tt + (idx >> 6u);
                let o = idx & 63u;
                var va: f32 = 0.0;
                var vb: f32 = 0.0;
                if (item < cfg.m) {
                    if (colBase + o < cfg.d) {
                        va = phi[item * cfg.d + colBase + o];
                    }
                    if (rowBase + o < cfg.r_rows && cfg.r_off + rowBase + o < cfg.d) {
                        vb = phi[item * cfg.d + cfg.r_off + rowBase + o];
                    }
                }
                wg_a[idx] = va;
                wg_b[idx] = vb;
            }
            workgroupBarrier();
            var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
            var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
            var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0;
            var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                let o = k * 64u;
                let a0 = wg_a[o + lid.x];
                let a1 = wg_a[o + lid.x + 16u];
                let a2 = wg_a[o + lid.x + 32u];
                let a3 = wg_a[o + lid.x + 48u];
                let b0 = wg_b[o + lid.y];
                let b1 = wg_b[o + lid.y + 16u];
                let b2 = wg_b[o + lid.y + 32u];
                let b3 = wg_b[o + lid.y + 48u];
                f0 = f0 + b0 * a0; f1 = f1 + b0 * a1; f2 = f2 + b0 * a2; f3 = f3 + b0 * a3;
                f4 = f4 + b1 * a0; f5 = f5 + b1 * a1; f6 = f6 + b1 * a2; f7 = f7 + b1 * a3;
                f8 = f8 + b2 * a0; f9 = f9 + b2 * a1; f10 = f10 + b2 * a2; f11 = f11 + b2 * a3;
                f12 = f12 + b3 * a0; f13 = f13 + b3 * a1; f14 = f14 + b3 * a2; f15 = f15 + b3 * a3;
            }
            let pb = j * 16u;
            pa[pb + 0u] = pa[pb + 0u] + f0;
            pa[pb + 1u] = pa[pb + 1u] + f1;
            pa[pb + 2u] = pa[pb + 2u] + f2;
            pa[pb + 3u] = pa[pb + 3u] + f3;
            pa[pb + 4u] = pa[pb + 4u] + f4;
            pa[pb + 5u] = pa[pb + 5u] + f5;
            pa[pb + 6u] = pa[pb + 6u] + f6;
            pa[pb + 7u] = pa[pb + 7u] + f7;
            pa[pb + 8u] = pa[pb + 8u] + f8;
            pa[pb + 9u] = pa[pb + 9u] + f9;
            pa[pb + 10u] = pa[pb + 10u] + f10;
            pa[pb + 11u] = pa[pb + 11u] + f11;
            pa[pb + 12u] = pa[pb + 12u] + f12;
            pa[pb + 13u] = pa[pb + 13u] + f13;
            pa[pb + 14u] = pa[pb + 14u] + f14;
            pa[pb + 15u] = pa[pb + 15u] + f15;
            workgroupBarrier();
        }
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = rowBase + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = colBase + lid.x + 16u * k2;
                if (c < cfg.d) {
                    let q = k * 4u + k2;
                    let ix = lr * cfg.d + c;
                    h[ix] = h[ix] + (pa[q] + pa[q + 16u]);
                }
            }
        }
    }
}
