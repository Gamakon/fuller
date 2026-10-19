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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) { acc[q & 15u] = 0.0; }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var n = 0u; n < 4u; n = n + 1u) {
            let e = tid + n * 256u;
            let k = e >> 6u;
            let i = e & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let lr = row0 + i;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
                let c = col0 + i;
                if (c < d) { vb = phi[t * d + c]; }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let a0 = sa[(o + ty) & 1023u];
            let a1 = sa[(o + ty + 16u) & 1023u];
            let a2 = sa[(o + ty + 32u) & 1023u];
            let a3 = sa[(o + ty + 48u) & 1023u];
            let b0 = sb[(o + tx) & 1023u];
            let b1 = sb[(o + tx + 16u) & 1023u];
            let b2 = sb[(o + tx + 32u) & 1023u];
            let b3 = sb[(o + tx + 48u) & 1023u];
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
