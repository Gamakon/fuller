// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative, coalesced tile load: 16 items x 64 columns for each operand
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let it = e >> 6u;
            let cc = e & 63u;
            let t = t0 + it;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (col0 + cc < d) {
                    vb = phi[base + col0 + cc];
                }
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) {
                    va = phi[base + cfg.r_off + row0 + cc];
                }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ro = kk * 64u + lid.y;
            let co = kk * 64u + lid.x;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = wg_a[(ro + i * 16u) & 1023u];
                b[i & 3u] = wg_b[(co + i * 16u) & 1023u];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[(i * 4u + j) & 15u] = acc[(i * 4u + j) & 15u] + a[i & 3u] * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i * 16u + lid.y;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + j * 16u + lid.x;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
