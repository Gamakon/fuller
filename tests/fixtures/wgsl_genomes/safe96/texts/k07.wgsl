// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 1024>;
var<workgroup> sB: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wg.x * 64u;
    let row0 = wg.y * 64u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = lid + 256u * k;
            let item = e >> 6u;
            let cc = e & 63u;
            let t = t0 + item;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                let gc = col0 + cc;
                if (gc < d) { vb = phi[base + gc]; }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va = phi[base + gr]; }
            }
            sA[e] = va;
            sB[e] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a0 = sA[o + ty];
            let a1 = sA[o + ty + 16u];
            let a2 = sA[o + ty + 32u];
            let a3 = sA[o + ty + 48u];
            let b0 = sB[o + tx];
            let b1 = sB[o + tx + 16u];
            let b2 = sB[o + tx + 32u];
            let b3 = sB[o + tx + 48u];
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
        t0 = t0 + 16u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
