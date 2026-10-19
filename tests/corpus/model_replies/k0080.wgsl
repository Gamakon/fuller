// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    var t0 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + 256u * k;
            let tt = e >> 6u;
            let cc = e & 63u;
            let t = t0 + tt;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let c = col0 + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            wa[e & 1023u] = va;
            wb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var tt = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = wa[(o + ty + 16u * i) & 1023u];
                b[i & 3u] = wb[(o + tx + 16u * i) & 1023u];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = acc[q] + a[i & 3u] * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
