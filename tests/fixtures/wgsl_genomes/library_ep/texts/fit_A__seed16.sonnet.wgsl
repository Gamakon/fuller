// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sc: array<f32, 512>;
var<workgroup> sr: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let cbase = wg.x * 64u;
    let rbase = wg.y * 64u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 8u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let e = l + 256u * k;
            let t = e >> 6u;
            let col = e & 63u;
            let tt = t0 + t;
            var vc: f32 = 0.0;
            var vr: f32 = 0.0;
            if (tt < cfg.m) {
                let c = cbase + col;
                if (c < cfg.d) { vc = phi[tt * cfg.d + c]; }
                let lr = rbase + col;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) { vr = phi[tt * cfg.d + gr]; }
            }
            sc[e] = vc;
            sr[e] = vr;
        }
        workgroupBarrier();
        for (var t = 0u; t < 8u; t = t + 1u) {
            let o = t * 64u;
            let c0 = sc[o + tx];
            let c1 = sc[o + tx + 16u];
            let c2 = sc[o + tx + 32u];
            let c3 = sc[o + tx + 48u];
            let r0 = sr[o + ty];
            let r1 = sr[o + ty + 16u];
            let r2 = sr[o + ty + 32u];
            let r3 = sr[o + ty + 48u];
            acc[0] = acc[0] + r0 * c0;
            acc[1] = acc[1] + r0 * c1;
            acc[2] = acc[2] + r0 * c2;
            acc[3] = acc[3] + r0 * c3;
            acc[4] = acc[4] + r1 * c0;
            acc[5] = acc[5] + r1 * c1;
            acc[6] = acc[6] + r1 * c2;
            acc[7] = acc[7] + r1 * c3;
            acc[8] = acc[8] + r2 * c0;
            acc[9] = acc[9] + r2 * c1;
            acc[10] = acc[10] + r2 * c2;
            acc[11] = acc[11] + r2 * c3;
            acc[12] = acc[12] + r3 * c0;
            acc[13] = acc[13] + r3 * c1;
            acc[14] = acc[14] + r3 * c2;
            acc[15] = acc[15] + r3 * c3;
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = cbase + tx + 16u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
