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
    var ra: array<f32, 8>;
    var rb: array<f32, 8>;
    for (var q = 0u; q < 64u; q = q + 1u) {
        acc[q & 63u] = 0.0;
    }
    let tid = lid.y * 16u + lid.x;
    let d = cfg.d;
    let col0 = wid.x * 128u;
    let row0 = wid.y * 128u;
    for (var t = 0u; t < cfg.m; t = t + 8u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let it = e >> 7u;
            let off = e & 127u;
            let item = t + it;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (item < cfg.m) {
                let base = item * d;
                let c = col0 + off;
                if (c < d) {
                    fb = phi[base + c];
                }
                let lr = row0 + off;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    fa = phi[base + cfg.r_off + lr];
                }
            }
            wg_a[e & 1023u] = fa;
            wg_b[e & 1023u] = fb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 8u; kk = kk + 1u) {
            for (var i = 0u; i < 8u; i = i + 1u) {
                ra[i & 7u] = wg_a[(kk * 128u + lid.y + i * 16u) & 1023u];
                rb[i & 7u] = wg_b[(kk * 128u + lid.x + i * 16u) & 1023u];
            }
            for (var i = 0u; i < 8u; i = i + 1u) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let ix = (i * 8u + j) & 63u;
                    acc[ix] = acc[ix] + ra[i & 7u] * rb[j & 7u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + lid.y + i * 16u;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                let c = col0 + lid.x + j * 16u;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
