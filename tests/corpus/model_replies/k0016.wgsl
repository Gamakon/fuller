// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 512>;
var<workgroup> wb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let colBase = wg.x * 64u;
    let rowBase = wg.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        // cooperative load, zero-filled beyond m / d / r_rows (handles remainder)
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let idx = l + 256u * e;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            let col = colBase + j;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                if (col < d) {
                    vb = phi[t * d + col];
                }
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            wb[idx & 511u] = vb;
            wa[idx & 511u] = va;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let o = k * 64u;
            let a0 = wa[(o + ty) & 511u];
            let a1 = wa[(o + ty + 16u) & 511u];
            let a2 = wa[(o + ty + 32u) & 511u];
            let a3 = wa[(o + ty + 48u) & 511u];
            let b0 = wb[(o + tx) & 511u];
            let b1 = wb[(o + tx + 16u) & 511u];
            let b2 = wb[(o + tx + 32u) & 511u];
            let b3 = wb[(o + tx + 48u) & 511u];
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
        let lr = rowBase + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx + 16u * j;
                if (c < d) {
                    let hi = lr * d + c;
                    h[hi] = h[hi] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
