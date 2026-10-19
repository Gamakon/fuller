// grid 16 16 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<f32, 64>;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) {
        acc[q & 63u] = 0.0;
    }
    let lidx = lid.y * 16u + lid.x;
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 128u;
    let row0 = wid.y * 128u;
    for (var t: u32 = 0u; t < m; t = t + 8u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = e >> 7u;
            let cc = e & 127u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + it < m) {
                let base = (t + it) * d;
                if (col0 + cc < d) {
                    fb = phi[base + col0 + cc];
                }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    fa = phi[base + cfg.r_off + lr];
                }
            }
            wg_a[e & 1023u] = fa;
            wg_b[e & 1023u] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            var av: array<f32, 8>;
            var bv: array<f32, 8>;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                av[i & 7u] = wg_a[(kk * 128u + i * 16u + lid.y) & 1023u];
                bv[i & 7u] = wg_b[(kk * 128u + i * 16u + lid.x) & 1023u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let a = av[i & 7u];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    let idx = (i * 8u + j) & 63u;
                    acc[idx] = fma(a, bv[j & 7u], acc[idx]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + i * 16u + lid.y;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let c = col0 + j * 16u + lid.x;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
