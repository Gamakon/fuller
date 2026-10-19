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
    var av: array<f32, 8>;
    var bv: array<f32, 8>;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) { acc[q & 63u] = 0.0; }
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 8u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + q * 256u;
            let kk = e >> 7u;
            let c = e & 127u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                let gc = wid.x * 128u + c;
                if (gc < d) { vb = phi[t * d + gc]; }
                let lr = wid.y * 128u + c;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let base = kk * 128u;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                av[i & 7u] = wg_a[(base + ly + 16u * i) & 1023u];
                bv[i & 7u] = wg_b[(base + lx + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let a = av[i & 7u];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    let ix = (i * 8u + j) & 63u;
                    acc[ix] = fma(a, bv[j & 7u], acc[ix]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 128u + ly + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let c = wid.x * 128u + lx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
