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
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let tk = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let gc = colBase + cc;
                if (gc < d) {
                    vb = phi[t * d + gc];
                }
                let lrow = rowBase + cc;
                let gr = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            wg_a[idx & 1023u] = va;
            wg_b[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a0 = wg_a[(o + lid.y) & 1023u];
            let a1 = wg_a[(o + 16u + lid.y) & 1023u];
            let a2 = wg_a[(o + 32u + lid.y) & 1023u];
            let a3 = wg_a[(o + 48u + lid.y) & 1023u];
            let b0 = wg_b[(o + lid.x) & 1023u];
            let b1 = wg_b[(o + 16u + lid.x) & 1023u];
            let b2 = wg_b[(o + 32u + lid.x) & 1023u];
            let b3 = wg_b[(o + 48u + lid.x) & 1023u];
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
        let lr = rowBase + i * 16u + lid.y;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + j * 16u + lid.x;
                if (c < d) {
                    let hi = lr * d + c;
                    h[hi] = h[hi] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
