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
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q & 15u] = 0.0; }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative, coalesced load: 16 items x 64 columns for each operand
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + it;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (t < m) {
                let c = col0 + cc;
                if (c < d) { fb = phi[t * d + c]; }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { fa = phi[t * d + gr]; }
            }
            wg_b[idx & 1023u] = fb;
            wg_a[idx & 1023u] = fa;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 16u; s = s + 1u) {
            let base = s * 64u;
            var av: array<f32, 4>;
            var bv: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                av[i & 3u] = wg_a[(base + lid.y + 16u * i) & 1023u];
                bv[i & 3u] = wg_b[(base + lid.x + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = fma(av[i & 3u], bv[j & 3u], acc[q]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
