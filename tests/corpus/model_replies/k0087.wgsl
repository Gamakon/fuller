// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 1024>;
var<workgroup> sB: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let col = e & 63u;
            let tt = e >> 6u;
            let t = t0 + tt;
            let lrow = rowBase + col;
            let ccol = colBase + col;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (lrow < cfg.r_rows && (cfg.r_off + lrow) < d) {
                    a = phi[base + cfg.r_off + lrow];
                }
                if (ccol < d) {
                    b = phi[base + ccol];
                }
            }
            sA[e & 1023u] = a;
            sB[e & 1023u] = b;
        }
        workgroupBarrier();

        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 64u;
            let a0 = sA[(o + ty) & 1023u];
            let a1 = sA[(o + ty + 16u) & 1023u];
            let a2 = sA[(o + ty + 32u) & 1023u];
            let a3 = sA[(o + ty + 48u) & 1023u];
            let b0 = sB[(o + tx) & 1023u];
            let b1 = sB[(o + tx + 16u) & 1023u];
            let b2 = sB[(o + tx + 32u) & 1023u];
            let b3 = sB[(o + tx + 48u) & 1023u];
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
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
