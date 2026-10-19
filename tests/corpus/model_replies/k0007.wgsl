// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = tid + e * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let lrr = row0 + j;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let cc = col0 + j;
                if (cc < d) {
                    vb = phi[t * d + cc];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();

        for (var k = 0u; k < 16u; k = k + 1u) {
            let ba = k * 64u + ty;
            let bb = k * 64u + tx;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var q = 0u; q < 4u; q = q + 1u) {
                a[q & 3u] = sa[(ba + q * 16u) & 1023u];
                b[q & 3u] = sb[(bb + q * 16u) & 1023u];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                let aj = a[j & 3u];
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let p = (j * 4u + i) & 15u;
                    acc[p] = acc[p] + aj * b[i & 3u];
                }
            }
        }
        workgroupBarrier();
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + ty + j * 16u;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = col0 + tx + i * 16u;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
