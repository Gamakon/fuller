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
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t = 0u; t < m; t = t + 16u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let e = lidx + 256u * j;
            let tt = e >> 6u;
            let cc = e & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t + tt < m) {
                let base = (t + tt) * d;
                let c = col0 + cc;
                if (c < d) {
                    va = phi[base + c];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[base + gr];
                }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = wg_a[(k * 64u + lid.x * 4u + i) & 1023u];
                b[i & 3u] = wg_b[(k * 64u + lid.y * 4u + i) & 1023u];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let idx = (j * 4u + i) & 15u;
                    acc[idx] = acc[idx] + b[j & 3u] * a[i & 3u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y * 4u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = col0 + lid.x * 4u + i;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
