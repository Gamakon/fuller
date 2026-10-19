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
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;
    let lx = lid3.x;
    let ly = lid3.y;
    let lid = ly * 16u + lx;
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var e: u32 = 0u; e < 4u; e = e + 1u) {
            let idx = lid + e * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            let col = c0 + j;
            let lr = r0 + j;
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
            sa[idx & 1023u] = va;
            sb[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ab = k * 64u + ly * 4u;
            let bb = k * 64u + lx * 4u;
            let a0 = sa[ab & 1023u];
            let a1 = sa[(ab + 1u) & 1023u];
            let a2 = sa[(ab + 2u) & 1023u];
            let a3 = sa[(ab + 3u) & 1023u];
            let b0 = sb[bb & 1023u];
            let b1 = sb[(bb + 1u) & 1023u];
            let b2 = sb[(bb + 2u) & 1023u];
            let b3 = sb[(bb + 3u) & 1023u];
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
        t0 = t0 + 16u;
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + ly * 4u + i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = c0 + lx * 4u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
