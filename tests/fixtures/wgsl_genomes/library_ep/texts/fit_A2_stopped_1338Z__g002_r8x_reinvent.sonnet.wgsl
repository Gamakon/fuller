// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        s[k] = 0.0;
    }
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = t + (idx >> 6u);
            let cc = idx & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tr < cfg.m) {
                if (colbase + cc < cfg.d) {
                    va = phi[tr * cfg.d + colbase + cc];
                }
                if (rowbase + cc < cfg.r_rows && cfg.r_off + rowbase + cc < cfg.d) {
                    vb = phi[tr * cfg.d + cfg.r_off + rowbase + cc];
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
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
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
            f0 = f0 + b0 * a0; f1 = f1 + b0 * a1;
            f2 = f2 + b1 * a0; f3 = f3 + b1 * a1;
            f4 = f4 + b2 * a0; f5 = f5 + b2 * a1;
            f6 = f6 + b3 * a0; f7 = f7 + b3 * a1;
            f8 = f8 + b4 * a0; f9 = f9 + b4 * a1;
            f10 = f10 + b5 * a0; f11 = f11 + b5 * a1;
            f12 = f12 + b6 * a0; f13 = f13 + b6 * a1;
            f14 = f14 + b7 * a0; f15 = f15 + b7 * a1;
        }
        s[0] = s[0] + f0; s[1] = s[1] + f1; s[2] = s[2] + f2; s[3] = s[3] + f3;
        s[4] = s[4] + f4; s[5] = s[5] + f5; s[6] = s[6] + f6; s[7] = s[7] + f7;
        s[8] = s[8] + f8; s[9] = s[9] + f9; s[10] = s[10] + f10; s[11] = s[11] + f11;
        s[12] = s[12] + f12; s[13] = s[13] + f13; s[14] = s[14] + f14; s[15] = s[15] + f15;
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let lr = rowbase + lid.y + 8u * j;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var c: u32 = 0u; c < 2u; c = c + 1u) {
                let col = colbase + lid.x + 32u * c;
                if (col < cfg.d) {
                    let ix = lr * cfg.d + col;
                    h[ix] = h[ix] + s[j * 2u + c];
                }
            }
        }
    }
}
