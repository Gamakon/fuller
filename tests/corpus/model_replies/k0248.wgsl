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
    for (var q: u32 = 0u; q < 64u; q = q + 1u) {
        acc[q & 63u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 128u;
    let row0 = wid.y * 128u;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e >> 7u;
            let cc = e & 127u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let col = col0 + cc;
                if (col < d) {
                    va = phi[t * d + col];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[t * d + gr];
                }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                ra[j & 7u] = wg_a[(kk * 128u + lid.x + 16u * j) & 1023u];
                rb[j & 7u] = wg_b[(kk * 128u + lid.y + 16u * j) & 1023u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let bv = rb[i & 7u];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    let idx = (i * 8u + j) & 63u;
                    acc[idx] = acc[idx] + bv * ra[j & 7u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
