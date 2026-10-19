// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative load: 16 items x 64 cols for B and A
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lid + k * 256u;
            let it = e >> 6u;
            let o = e & 63u;
            let t = t0 + it;
            let c = col0 + o;
            let lr = row0 + o;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (c < d) { vb = phi[base + c]; }
                if (lr < cfg.r_rows && gr < d) { va = phi[base + gr]; }
            }
            sb[e & 1023u] = vb;
            sa[e & 1023u] = va;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 4u) {
            for (var u: u32 = 0u; u < 4u; u = u + 1u) {
                let o = (kk + u) * 64u;
                let a0 = sa[(o + ty * 4u) & 1023u];
                let a1 = sa[(o + ty * 4u + 1u) & 1023u];
                let a2 = sa[(o + ty * 4u + 2u) & 1023u];
                let a3 = sa[(o + ty * 4u + 3u) & 1023u];
                let b0 = sb[(o + tx * 4u) & 1023u];
                let b1 = sb[(o + tx * 4u + 1u) & 1023u];
                let b2 = sb[(o + tx * 4u + 2u) & 1023u];
                let b3 = sb[(o + tx * 4u + 3u) & 1023u];
                acc[0u] = acc[0u] + a0 * b0;
                acc[1u] = acc[1u] + a0 * b1;
                acc[2u] = acc[2u] + a0 * b2;
                acc[3u] = acc[3u] + a0 * b3;
                acc[4u] = acc[4u] + a1 * b0;
                acc[5u] = acc[5u] + a1 * b1;
                acc[6u] = acc[6u] + a1 * b2;
                acc[7u] = acc[7u] + a1 * b3;
                acc[8u] = acc[8u] + a2 * b0;
                acc[9u] = acc[9u] + a2 * b1;
                acc[10u] = acc[10u] + a2 * b2;
                acc[11u] = acc[11u] + a2 * b3;
                acc[12u] = acc[12u] + a3 * b0;
                acc[13u] = acc[13u] + a3 * b1;
                acc[14u] = acc[14u] + a3 * b2;
                acc[15u] = acc[15u] + a3 * b3;
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
