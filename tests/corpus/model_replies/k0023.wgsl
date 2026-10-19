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
    let x0 = wid.x * 64u;
    let y0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lin + 256u * k;
            let kk = e >> 6u;
            let cc = e & 63u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let col = x0 + cc;
                if (col < d) { vb = phi[t * d + col]; }
                let lr = y0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let base = kk * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sa[(base + ty + 16u * i) & 1023u];
                b[i & 3u] = sb[(base + tx + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let ai = a[i & 3u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = acc[q] + ai * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = y0 + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = x0 + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
