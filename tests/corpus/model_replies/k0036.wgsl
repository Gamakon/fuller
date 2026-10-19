// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let bx0 = wid.x * 64u;
    let by0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q = 0u; q < 4u; q = q + 1u) {
            let idx = tid + 256u * q;
            let k = idx >> 6u;
            let i = idx & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let lr = by0 + i;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = bx0 + i;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < 16u; kk = kk + 4u) {
            for (var u = 0u; u < 4u; u = u + 1u) {
                let base = (kk + u) * 64u;
                var a: array<f32, 4>;
                var b: array<f32, 4>;
                for (var i = 0u; i < 4u; i = i + 1u) {
                    a[i & 3u] = sa[(base + ty + 16u * i) & 1023u];
                    b[i & 3u] = sb[(base + tx + 16u * i) & 1023u];
                }
                for (var i = 0u; i < 4u; i = i + 1u) {
                    for (var j = 0u; j < 4u; j = j + 1u) {
                        let n = (i * 4u + j) & 15u;
                        acc[n] = fma(a[i & 3u], b[j & 3u], acc[n]);
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = by0 + ty + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = bx0 + tx + 16u * j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
