// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let li = lid.y * 16u + lid.x;
    var acc: array<f32, 16>;
    var tmp: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { acc[k] = 0.0; }
    let d = cfg.d;
    for (var t = 0u; t < cfg.m; t = t + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = li + 256u * k;
            let tt = t + (idx >> 6u);
            let cc = idx & 63u;
            let gc = wid.x * 64u + cc;
            let lr = wid.y * 64u + cc;
            var fa = 0.0;
            var fb = 0.0;
            if (tt < cfg.m) {
                if (gc < d) { fb = phi[tt * d + gc]; }
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { fa = phi[tt * d + cfg.r_off + lr]; }
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) { tmp[k] = 0.0; }
        for (var k = 0u; k < 16u; k = k + 1u) {
            var pb: array<f32, 4>;
            var pc: array<f32, 4>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                pb[j] = wg_a[k * 64u + lid.y + 16u * j];
                pc[j] = wg_b[k * 64u + lid.x + 16u * j];
            }
            for (var a = 0u; a < 4u; a = a + 1u) {
                for (var b = 0u; b < 4u; b = b + 1u) {
                    tmp[a * 4u + b] = tmp[a * 4u + b] + pb[a] * pc[b];
                }
            }
        }
        for (var k = 0u; k < 16u; k = k + 1u) { acc[k] = acc[k] + tmp[k]; }
        workgroupBarrier();
    }
    for (var a = 0u; a < 4u; a = a + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * a;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var b = 0u; b < 4u; b = b + 1u) {
                let c = wid.x * 64u + lid.x + 16u * b;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[a * 4u + b];
                }
            }
        }
    }
}
