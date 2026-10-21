// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let l = ly * 16u + lx;
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = l + i * 256u;
            let kk = e >> 6u;
            let j = e & 63u;
            let t = t0 + kk;
            var av = 0.0;
            var bv = 0.0;
            if (t < m) {
                let lrow = row0 + j;
                let gr = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            As[e] = av;
            Bs[e] = bv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let a0 = As[o + ly];
            let a1 = As[o + ly + 16u];
            let a2 = As[o + ly + 32u];
            let a3 = As[o + ly + 48u];
            let b0 = Bs[o + lx];
            let b1 = Bs[o + lx + 16u];
            let b2 = Bs[o + lx + 32u];
            let b3 = Bs[o + lx + 48u];
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
        let lrow = row0 + ly + i * 16u;
        let gr = cfg.r_off + lrow;
        if (lrow < cfg.r_rows && gr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lx + j * 16u;
                if (c < d) {
                    let idx = lrow * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
