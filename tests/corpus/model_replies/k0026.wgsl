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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load of 16 items x 64 rows and 16 items x 64 cols
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrw = row0 + j;
                let gr = cfg.r_off + lrw;
                if (lrw < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 4u) {
            for (var u: u32 = 0u; u < 4u; u = u + 1u) {
                let base = (kk + u) * 64u;
                let a0 = sa[(base + ty) & 1023u];
                let a1 = sa[(base + ty + 16u) & 1023u];
                let a2 = sa[(base + ty + 32u) & 1023u];
                let a3 = sa[(base + ty + 48u) & 1023u];
                let b0 = sb[(base + tx) & 1023u];
                let b1 = sb[(base + tx + 16u) & 1023u];
                let b2 = sb[(base + tx + 32u) & 1023u];
                let b3 = sb[(base + tx + 48u) & 1023u];
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
        }
        workgroupBarrier();
    }

    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + ty + 16u * j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = col0 + tx + 16u * i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
