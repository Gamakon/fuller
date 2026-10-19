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
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = e >> 6u;
            let cc = e & 63u;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (t + it < m) {
                let base = (t + it) * d;
                if (col0 + cc < d) {
                    fb = phi[base + col0 + cc];
                }
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) {
                    fa = phi[base + cfg.r_off + row0 + cc];
                }
            }
            wg_a[e & 1023u] = fa;
            wg_b[e & 1023u] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            var bv: array<f32, 4>;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                bv[j & 3u] = wg_b[(o + lid.x + 16u * j) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let av = wg_a[(o + lid.y + 16u * i) & 1023u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let ix = (i * 4u + j) & 15u;
                    acc[ix] = fma(av, bv[j & 3u], acc[ix]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
