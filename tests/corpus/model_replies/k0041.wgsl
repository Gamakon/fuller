// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 32u;
    let row0 = wid.y * 32u;

    var acc: array<f32, 4>;
    acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0; acc[3] = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = tid + 256u * k;
            let tt = e >> 5u;
            let j = e & 31u;
            let t = t0 + tt;
            let lrr = row0 + j;
            let grr = cfg.r_off + lrr;
            let cc = col0 + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lrr < cfg.r_rows && grr < d) {
                    va = phi[t * d + grr];
                }
                if (cc < d) {
                    vb = phi[t * d + cc];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let a0 = sa[(o + ly) & 511u];
            let a1 = sa[(o + ly + 16u) & 511u];
            let b0 = sb[(o + lx) & 511u];
            let b1 = sb[(o + lx + 16u) & 511u];
            acc[0] = acc[0] + a0 * b0;
            acc[1] = acc[1] + a0 * b1;
            acc[2] = acc[2] + a1 * b0;
            acc[3] = acc[3] + a1 * b1;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let lr = row0 + ly + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let c = col0 + lx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 2u + j) & 3u];
                }
            }
        }
    }
}
