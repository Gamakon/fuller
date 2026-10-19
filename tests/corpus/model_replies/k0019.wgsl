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
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var e: u32 = 0u; e < 4u; e = e + 1u) {
            let idx = tid + 256u * e;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = rowBase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colBase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a0 = sa[(o + ty) & 1023u];
            let a1 = sa[(o + ty + 16u) & 1023u];
            let a2 = sa[(o + ty + 32u) & 1023u];
            let a3 = sa[(o + ty + 48u) & 1023u];
            let b0 = sb[(o + tx) & 1023u];
            let b1 = sb[(o + tx + 16u) & 1023u];
            let b2 = sb[(o + tx + 32u) & 1023u];
            let b3 = sb[(o + tx + 48u) & 1023u];
            acc[0] = fma(a0, b0, acc[0]);
            acc[1] = fma(a0, b1, acc[1]);
            acc[2] = fma(a0, b2, acc[2]);
            acc[3] = fma(a0, b3, acc[3]);
            acc[4] = fma(a1, b0, acc[4]);
            acc[5] = fma(a1, b1, acc[5]);
            acc[6] = fma(a1, b2, acc[6]);
            acc[7] = fma(a1, b3, acc[7]);
            acc[8] = fma(a2, b0, acc[8]);
            acc[9] = fma(a2, b1, acc[9]);
            acc[10] = fma(a2, b2, acc[10]);
            acc[11] = fma(a2, b3, acc[11]);
            acc[12] = fma(a3, b0, acc[12]);
            acc[13] = fma(a3, b1, acc[13]);
            acc[14] = fma(a3, b2, acc[14]);
            acc[15] = fma(a3, b3, acc[15]);
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx + 16u * j;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
