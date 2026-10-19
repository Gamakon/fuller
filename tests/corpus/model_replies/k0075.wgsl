// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> tA: array<f32, 512>;
var<workgroup> tB: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let cb = wid.x * 32u;
    let rb = wid.y * 32u;

    var acc: array<f32, 8>;
    for (var q: u32 = 0u; q < 8u; q = q + 1u) {
        acc[q & 7u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = lin + k * 256u;
            let it = e >> 5u;
            let cc = e & 31u;
            let t = t0 + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrw = rb + cc;
                let gr = cfg.r_off + lrw;
                if (lrw < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = cb + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            tA[e & 511u] = va;
            tB[e & 511u] = vb;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let i0 = (2u * kk) * 32u;
            let i1 = (2u * kk + 1u) * 32u;
            let a00 = tA[(i0 + ty) & 511u];
            let a01 = tA[(i0 + ty + 16u) & 511u];
            let b00 = tB[(i0 + tx) & 511u];
            let b01 = tB[(i0 + tx + 16u) & 511u];
            let a10 = tA[(i1 + ty) & 511u];
            let a11 = tA[(i1 + ty + 16u) & 511u];
            let b10 = tB[(i1 + tx) & 511u];
            let b11 = tB[(i1 + tx + 16u) & 511u];
            // set 0: even items -> acc[0..3], set 1: odd items -> acc[4..7]
            acc[0u] = acc[0u] + a00 * b00;
            acc[1u] = acc[1u] + a00 * b01;
            acc[2u] = acc[2u] + a01 * b00;
            acc[3u] = acc[3u] + a01 * b01;
            acc[4u] = acc[4u] + a10 * b10;
            acc[5u] = acc[5u] + a10 * b11;
            acc[6u] = acc[6u] + a11 * b10;
            acc[7u] = acc[7u] + a11 * b11;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let lr = rb + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let c = cb + tx + 16u * j;
                if (c < d) {
                    let q = i * 2u + j;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (acc[q & 7u] + acc[(q + 4u) & 7u]);
                }
            }
        }
    }
}
