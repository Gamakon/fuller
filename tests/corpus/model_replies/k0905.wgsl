// grid 8 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 64>;
    var ra: array<f32, 8>;
    var rb: array<f32, 8>;
    for (var q = 0u; q < 64u; q = q + 1u) {
        acc[q & 63u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k = 0u; k < 16u; k = k + 1u) {
            let idx = lidx + k * 64u;
            let it = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + it;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t < m) {
                let col = colBase + cc;
                if (col < d) {
                    fb = phi[t * d + col];
                }
                let row = rowBase + cc;
                if (row < cfg.r_rows && (cfg.r_off + row) < d) {
                    fa = phi[t * d + cfg.r_off + row];
                }
            }
            wg_a[idx & 1023u] = fa;
            wg_b[idx & 1023u] = fb;
        }
        workgroupBarrier();
        for (var tt = 0u; tt < 16u; tt = tt + 1u) {
            for (var i = 0u; i < 8u; i = i + 1u) {
                ra[i & 7u] = wg_a[(tt * 64u + lid.y + 8u * i) & 1023u];
                rb[i & 7u] = wg_b[(tt * 64u + lid.x + 8u * i) & 1023u];
            }
            for (var i = 0u; i < 8u; i = i + 1u) {
                let av = ra[i & 7u];
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let ai = (i * 8u + j) & 63u;
                    acc[ai] = fma(av, rb[j & 7u], acc[ai]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        let row = rowBase + lid.y + 8u * i;
        if (row < cfg.r_rows && (cfg.r_off + row) < d) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                let col = colBase + lid.x + 8u * j;
                if (col < d) {
                    let idx = row * d + col;
                    h[idx] = h[idx] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
