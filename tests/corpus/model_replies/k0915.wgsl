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
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let item = idx >> 6u;
            let cc = idx & 63u;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (t + item < m) {
                let base = (t + item) * d;
                let gc = colBase + cc;
                if (gc < d) {
                    fb = phi[base + gc];
                }
                let lr = rowBase + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    fa = phi[base + cfg.r_off + lr];
                }
            }
            wg_a[idx & 1023u] = fa;
            wg_b[idx & 1023u] = fb;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let ob = tt * 64u;
            let a0 = wg_a[(ob + lid.y * 4u) & 1023u];
            let a1 = wg_a[(ob + lid.y * 4u + 1u) & 1023u];
            let a2 = wg_a[(ob + lid.y * 4u + 2u) & 1023u];
            let a3 = wg_a[(ob + lid.y * 4u + 3u) & 1023u];
            let b0 = wg_b[(ob + lid.x * 4u) & 1023u];
            let b1 = wg_b[(ob + lid.x * 4u + 1u) & 1023u];
            let b2 = wg_b[(ob + lid.x * 4u + 2u) & 1023u];
            let b3 = wg_b[(ob + lid.x * 4u + 3u) & 1023u];
            acc[0u] = fma(a0, b0, acc[0u]);
            acc[1u] = fma(a0, b1, acc[1u]);
            acc[2u] = fma(a0, b2, acc[2u]);
            acc[3u] = fma(a0, b3, acc[3u]);
            acc[4u] = fma(a1, b0, acc[4u]);
            acc[5u] = fma(a1, b1, acc[5u]);
            acc[6u] = fma(a1, b2, acc[6u]);
            acc[7u] = fma(a1, b3, acc[7u]);
            acc[8u] = fma(a2, b0, acc[8u]);
            acc[9u] = fma(a2, b1, acc[9u]);
            acc[10u] = fma(a2, b2, acc[10u]);
            acc[11u] = fma(a2, b3, acc[11u]);
            acc[12u] = fma(a3, b0, acc[12u]);
            acc[13u] = fma(a3, b1, acc[13u]);
            acc[14u] = fma(a3, b2, acc[14u]);
            acc[15u] = fma(a3, b3, acc[15u]);
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x * 4u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
