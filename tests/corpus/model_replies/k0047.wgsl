// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + 256u * k;
            let t = e >> 6u;
            let c = e & 63u;
            let it = t0 + t;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < m) {
                let base = it * d;
                let lr = rbase + c;
                let gr = r_off + lr;
                if (lr < r_rows && gr < d) {
                    va = phi[base + gr];
                }
                let gc = cbase + c;
                if (gc < d) {
                    vb = phi[base + gc];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let a0 = sa[(o + ty) & 1023u];
            let a1 = sa[(o + ty + 16u) & 1023u];
            let a2 = sa[(o + ty + 32u) & 1023u];
            let a3 = sa[(o + ty + 48u) & 1023u];
            let b0 = sb[(o + tx) & 1023u];
            let b1 = sb[(o + tx + 16u) & 1023u];
            let b2 = sb[(o + tx + 32u) & 1023u];
            let b3 = sb[(o + tx + 48u) & 1023u];
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
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + ty + 16u * i;
        if (lr < r_rows && r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cbase + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
