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
    var ra: array<f32, 8>;
    var rb: array<f32, 8>;
    for (var q = 0u; q < 64u; q = q + 1u) { acc[q & 63u] = 0.0; }
    let d = cfg.d;
    let colBase = wid.x * 128u;
    let rowBase = wid.y * 128u;
    for (var t = 0u; t < cfg.m; t = t + 8u) {
        workgroupBarrier();
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = e >> 7u;
            let c = e & 127u;
            let item = t + it;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (item < cfg.m) {
                let gc = colBase + c;
                if (gc < d) { fb = phi[item * d + gc]; }
                let lr = rowBase + c;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    fa = phi[item * d + cfg.r_off + lr];
                }
            }
            wg_a[e & 1023u] = fa;
            wg_b[e & 1023u] = fb;
        }
        workgroupBarrier();
        for (var i = 0u; i < 8u; i = i + 1u) {
            let o = i * 128u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                ra[j & 7u] = wg_a[(o + lid.y * 4u + j) & 1023u];
                ra[(j + 4u) & 7u] = wg_a[(o + 64u + lid.y * 4u + j) & 1023u];
                rb[j & 7u] = wg_b[(o + lid.x * 4u + j) & 1023u];
                rb[(j + 4u) & 7u] = wg_b[(o + 64u + lid.x * 4u + j) & 1023u];
            }
            for (var p = 0u; p < 8u; p = p + 1u) {
                let av = ra[p & 7u];
                for (var q = 0u; q < 8u; q = q + 1u) {
                    let ix = (p * 8u + q) & 63u;
                    acc[ix] = fma(av, rb[q & 7u], acc[ix]);
                }
            }
        }
    }
    for (var p = 0u; p < 8u; p = p + 1u) {
        let rl = rowBase + lid.y * 4u + (p & 3u) + (p >> 2u) * 64u;
        if (rl < cfg.r_rows && (cfg.r_off + rl) < d) {
            for (var q = 0u; q < 8u; q = q + 1u) {
                let cc = colBase + lid.x * 4u + (q & 3u) + (q >> 2u) * 64u;
                if (cc < d) {
                    let idx = rl * d + cc;
                    h[idx] = h[idx] + acc[(p * 8u + q) & 63u];
                }
            }
        }
    }
}
