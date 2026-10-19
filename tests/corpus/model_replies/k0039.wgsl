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
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = tid + e * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let col = cb + j;
                if (col < d) {
                    vb = phi[t * d + col];
                }
                let lrr = rb + j;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let ao = k * 64u + ly * 4u;
            let bo = k * 64u + lx * 4u;
            let a0 = sa[ao & 1023u];
            let a1 = sa[(ao + 1u) & 1023u];
            let a2 = sa[(ao + 2u) & 1023u];
            let a3 = sa[(ao + 3u) & 1023u];
            let b0 = sb[bo & 1023u];
            let b1 = sb[(bo + 1u) & 1023u];
            let b2 = sb[(bo + 2u) & 1023u];
            let b3 = sb[(bo + 3u) & 1023u];
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
        let lr = rb + ly * 4u + i;
        if (lr < cfg.r_rows) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = cb + lx * 4u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
