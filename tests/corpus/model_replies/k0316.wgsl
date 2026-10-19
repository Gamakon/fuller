// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let rbase = wg.y * 32u;
    let cbase = wg.x * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 1u; k = k + 1u) {
            // 256 entries per array: 8 items x 32 columns; one each per thread
            let t = l / 32u;
            let j = l % 32u;
            let tg = t0 + t;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tg < m) {
                let lrr = rbase + j;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    va = phi[tg * d + gr];
                }
                let cc = cbase + j;
                if (cc < d) {
                    vb = phi[tg * d + cc];
                }
            }
            sa[t * 32u + j] = va;
            sb[t * 32u + j] = vb;
        }
        workgroupBarrier();
        for (var t: u32 = 0u; t < 8u; t = t + 1u) {
            let r0 = sa[t * 32u + ty];
            let r1 = sa[t * 32u + ty + 16u];
            let c0 = sb[t * 32u + tx];
            let c1 = sb[t * 32u + tx + 16u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let lr = rbase + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let c = cbase + tx + 16u * j;
                if (c < d) {
                    var v: f32;
                    if (i == 0u) {
                        if (j == 0u) { v = a00; } else { v = a01; }
                    } else {
                        if (j == 0u) { v = a10; } else { v = a11; }
                    }
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v;
                }
            }
        }
    }
}
