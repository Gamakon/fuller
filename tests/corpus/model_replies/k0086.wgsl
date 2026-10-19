// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    // per-thread load slots: two (item, col) pairs for each tile
    let lc = tid & 63u;
    let li = tid >> 6u;          // 0..3, second slot is li+4
    let cc = col0 + lc;
    let lrw = row0 + lc;
    let grw = cfg.r_off + lrw;
    let colOk = cc < d;
    let rowOk = (lrw < cfg.r_rows) && (grw < d);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // load 8 items
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let k = li + s * 4u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowOk) { va = phi[base + grw]; }
                if (colOk) { vb = phi[base + cc]; }
            }
            sa[(k * 64u + lc) & 511u] = va;
            sb[(k * 64u + lc) & 511u] = vb;
        }
        workgroupBarrier();
        // unrolled 8 items
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let ab = k * 64u + lid.y * 4u;
            let bb = k * 64u + lid.x * 4u;
            let a0 = sa[ab & 511u];
            let a1 = sa[(ab + 1u) & 511u];
            let a2 = sa[(ab + 2u) & 511u];
            let a3 = sa[(ab + 3u) & 511u];
            let b0 = sb[bb & 511u];
            let b1 = sb[(bb + 1u) & 511u];
            let b2 = sb[(bb + 2u) & 511u];
            let b3 = sb[(bb + 3u) & 511u];
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
        t0 = t0 + 8u;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
