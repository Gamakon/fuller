// grid 16 16 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 64>;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) {
        acc[q & 63u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 128u;
    let rowBase = wid.y * 128u;
    for (var t: u32 = 0u; t < m; t = t + 8u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = e >> 7u;
            let cc = e & 127u;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (t + it < m) {
                let base = (t + it) * d;
                if (colBase + cc < d) {
                    fb = phi[base + colBase + cc];
                }
                if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < d) {
                    fa = phi[base + cfg.r_off + rowBase + cc];
                }
            }
            wg_a[e & 1023u] = fa;
            wg_b[e & 1023u] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            var ra: array<f32, 8>;
            var rb: array<f32, 8>;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                ra[i & 7u] = wg_a[(kk * 128u + lid.y * 8u + i) & 1023u];
                rb[i & 7u] = wg_b[(kk * 128u + lid.x * 8u + i) & 1023u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let av = ra[i & 7u];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    let idx = (i * 8u + j) & 63u;
                    acc[idx] = fma(av, rb[j & 7u], acc[idx]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = rowBase + lid.y * 8u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let c = colBase + lid.x * 8u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
