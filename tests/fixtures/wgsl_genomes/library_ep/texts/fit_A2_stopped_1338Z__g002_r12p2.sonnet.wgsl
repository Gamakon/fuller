// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var fa: f32 = 0.0; var fb: f32 = 0.0;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let idx = lid.y * 16u + lid.x + 256u * k;
            let it = t + (idx >> 6u);
            let cc = idx & 63u;
            fa = 0.0;
            fb = 0.0;
            if (it < cfg.m) {
                if (wid.x * 64u + cc < cfg.d) {
                    fa = phi[it * cfg.d + wid.x * 64u + cc];
                }
                if (wid.y * 64u + cc < cfg.r_rows && cfg.r_off + wid.y * 64u + cc < cfg.d) {
                    fb = phi[it * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            f0 = wg_b[k * 64u + lid.y];
            f1 = wg_b[k * 64u + lid.y + 16u];
            f2 = wg_b[k * 64u + lid.y + 32u];
            f3 = wg_b[k * 64u + lid.y + 48u];
            f4 = wg_a[k * 64u + lid.x];
            f5 = wg_a[k * 64u + lid.x + 16u];
            f6 = wg_a[k * 64u + lid.x + 32u];
            f7 = wg_a[k * 64u + lid.x + 48u];
            pa[0] = pa[0] + f0 * f4;
            pa[1] = pa[1] + f0 * f5;
            pa[2] = pa[2] + f0 * f6;
            pa[3] = pa[3] + f0 * f7;
            pa[4] = pa[4] + f1 * f4;
            pa[5] = pa[5] + f1 * f5;
            pa[6] = pa[6] + f1 * f6;
            pa[7] = pa[7] + f1 * f7;
            pa[8] = pa[8] + f2 * f4;
            pa[9] = pa[9] + f2 * f5;
            pa[10] = pa[10] + f2 * f6;
            pa[11] = pa[11] + f2 * f7;
            pa[12] = pa[12] + f3 * f4;
            pa[13] = pa[13] + f3 * f5;
            pa[14] = pa[14] + f3 * f6;
            pa[15] = pa[15] + f3 * f7;
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let lr = wid.y * 64u + lid.y + 16u * k;
            let c = wid.x * 64u + lid.x + 16u * k2;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 4u + k2];
            }
        }
    }
}
