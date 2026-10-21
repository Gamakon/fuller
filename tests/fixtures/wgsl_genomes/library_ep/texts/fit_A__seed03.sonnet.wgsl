// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 512>;
var<workgroup> Bs: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 8u + lid.x;
    let row0 = wid.y * 32u;   // local row base
    let col0 = wid.x * 32u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let nchunks = (m + 15u) / 16u;
    for (var ch = 0u; ch < nchunks; ch = ch + 1u) {
        let t0 = ch * 16u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let i = tid + j * 64u;
            let k = i >> 5u;
            let x = i & 31u;
            let t = t0 + k;
            let lr = row0 + x;
            let gr = cfg.r_off + lr;
            let c = col0 + x;
            var a = 0.0;
            var b = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) { a = phi[t * d + gr]; }
                if (c < d) { b = phi[t * d + c]; }
            }
            As[i] = a;
            Bs[i] = b;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let a0 = As[o + lid.y * 4u + 0u];
            let a1 = As[o + lid.y * 4u + 1u];
            let a2 = As[o + lid.y * 4u + 2u];
            let a3 = As[o + lid.y * 4u + 3u];
            let b0 = Bs[o + lid.x * 4u + 0u];
            let b1 = Bs[o + lid.x * 4u + 1u];
            let b2 = Bs[o + lid.x * 4u + 2u];
            let b3 = Bs[o + lid.x * 4u + 3u];
            acc[0] = acc[0] + a0 * b0; acc[1] = acc[1] + a0 * b1;
            acc[2] = acc[2] + a0 * b2; acc[3] = acc[3] + a0 * b3;
            acc[4] = acc[4] + a1 * b0; acc[5] = acc[5] + a1 * b1;
            acc[6] = acc[6] + a1 * b2; acc[7] = acc[7] + a1 * b3;
            acc[8] = acc[8] + a2 * b0; acc[9] = acc[9] + a2 * b1;
            acc[10] = acc[10] + a2 * b2; acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0; acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2; acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
