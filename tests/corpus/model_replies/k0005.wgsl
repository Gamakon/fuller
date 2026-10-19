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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 32>;
    for (var i = 0u; i < 32u; i = i + 1u) {
        acc[i & 31u] = 0.0;
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let t = e >> 6u;
            let j = e & 63u;
            let item = t0 + t;
            var va = 0.0;
            var vb = 0.0;
            if (item < m) {
                let c = col0 + j;
                if (c < d) {
                    vb = phi[item * d + c];
                }
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[item * d + gr];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var tt = 0u; tt < 16u; tt = tt + 2u) {
            for (var s = 0u; s < 2u; s = s + 1u) {
                let base = (tt + s) * 64u;
                var a: array<f32, 4>;
                var b: array<f32, 4>;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    a[q & 3u] = sa[(base + ty + 16u * q) & 1023u];
                    b[q & 3u] = sb[(base + tx + 16u * q) & 1023u];
                }
                let o = s * 16u;
                for (var j = 0u; j < 4u; j = j + 1u) {
                    for (var i = 0u; i < 4u; i = i + 1u) {
                        let ix = (o + j * 4u + i) & 31u;
                        acc[ix] = acc[ix] + a[j & 3u] * b[i & 3u];
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + ty + 16u * j;
        if (lr < cfg.r_rows) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = col0 + tx + 16u * i;
                if (c < d) {
                    let idx = lr * d + c;
                    let n = (j * 4u + i) & 31u;
                    h[idx] = h[idx] + (acc[n] + acc[(n + 16u) & 31u]);
                }
            }
        }
    }
}
