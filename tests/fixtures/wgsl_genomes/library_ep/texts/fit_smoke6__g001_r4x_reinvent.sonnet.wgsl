// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }
    let d = cfg.d;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 256u * k;
            let kk = e >> 6u;
            let cc = e & 63u;
            let tt = t + kk;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (tt < cfg.m) {
                let gc = col0 + cc;
                if (gc < d) { fb = phi[tt * d + gc]; }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) { fa = phi[tt * d + cfg.r_off + lr]; }
            }
            wg_a[e] = fa;
            wg_b[e] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i] = wg_a[k * 64u + lid.y * 4u + i];
                b[i] = wg_b[k * 64u + lid.x * 4u + i];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
