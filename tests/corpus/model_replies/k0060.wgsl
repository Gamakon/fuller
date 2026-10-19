// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let c0 = wid.x * 64u;
    let lr0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let e = tid + i * 256u;
            let kk = e >> 6u;
            let j = e & 63u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = lr0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = c0 + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let base = kk * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                a[q & 3u] = sa[(base + ty + 16u * q) & 511u];
                b[q & 3u] = sb[(base + tx + 16u * q) & 511u];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let k = (j * 4u + i) & 15u;
                    acc[k] = acc[k] + a[j & 3u] * b[i & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = lr0 + ty + 16u * j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = c0 + tx + 16u * i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
