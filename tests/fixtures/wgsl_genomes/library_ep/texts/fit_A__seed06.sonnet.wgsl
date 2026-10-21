// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 512>;
var<workgroup> Bs: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wg.x * 64u;
    let rowBase = wg.y * 64u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // cooperative tile load, zero-padded for item remainder / out-of-range
        for (var q = 0u; q < 2u; q = q + 1u) {
            let e = lid + 256u * q;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            let c = colBase + j;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) { av = phi[t * d + gr]; }
                if (c < d) { bv = phi[t * d + c]; }
            }
            As[e] = av;
            Bs[e] = bv;
        }
        workgroupBarrier();
        for (var k = 0u; k < 8u; k = k + 1u) {
            let o = k * 64u;
            let a0 = As[o + ty];
            let a1 = As[o + ty + 16u];
            let a2 = As[o + ty + 32u];
            let a3 = As[o + ty + 48u];
            let b0 = Bs[o + tx];
            let b1 = Bs[o + tx + 16u];
            let b2 = Bs[o + tx + 32u];
            let b3 = Bs[o + tx + 48u];
            acc[0] = acc[0] + a0 * b0;
            acc[1] = acc[1] + a0 * b1;
            acc[2] = acc[2] + a0 * b2;
            acc[3] = acc[3] + a0 * b3;
            acc[4] = acc[4] + a1 * b0;
            acc[5] = acc[5] + a1 * b1;
            acc[6] = acc[6] + a1 * b2;
            acc[7] = acc[7] + a1 * b3;
            acc[8] = acc[8] + a2 * b0;
            acc[9] = acc[9] + a2 * b1;
            acc[10] = acc[10] + a2 * b2;
            acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0;
            acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2;
            acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
