// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let col0 = wg.x * 64u;
    let row0 = wg.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        // cooperative load: 512 elements each for A and B, 2 per thread per... 
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = tid + 256u * k;
            let item = idx >> 6u;
            let off = idx & 63u;
            let t = t0 + item;
            let c = col0 + off;
            let lrow = row0 + off;
            let gr = r_off + lrow;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (c < d) {
                    vb = phi[base + c];
                }
                if (lrow < r_rows && gr < d) {
                    va = phi[base + gr];
                }
            }
            sb[idx & 511u] = vb;
            sa[idx & 511u] = va;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let o = kk * 64u;
            let a0 = sa[(o + ly) & 511u];
            let a1 = sa[(o + ly + 16u) & 511u];
            let a2 = sa[(o + ly + 32u) & 511u];
            let a3 = sa[(o + ly + 48u) & 511u];
            let b0 = sb[(o + lx) & 511u];
            let b1 = sb[(o + lx + 16u) & 511u];
            let b2 = sb[(o + lx + 32u) & 511u];
            let b3 = sb[(o + lx + 48u) & 511u];
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

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ly + 16u * i;
        if (lr < r_rows && (r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
