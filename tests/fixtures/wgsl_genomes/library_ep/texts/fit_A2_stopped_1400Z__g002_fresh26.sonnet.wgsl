// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let c0 = wid.x * 64u;
    let r0 = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let idx = lidx + 256u * j;
            let kk = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let col = c0 + cc;
                if (col < d) { va = phi[t * d + col]; }
                let lr = r0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { vb = phi[t * d + gr]; }
            }
            sa[idx] = va;
            sb[idx] = vb;
        }
        workgroupBarrier();
        var tmp: array<f32, 16>;
        for (var q: u32 = 0u; q < 16u; q = q + 1u) { tmp[q] = 0.0; }
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let base = kk * 64u;
            let a0 = sa[base + lid.x];
            let a1 = sa[base + lid.x + 16u];
            let a2 = sa[base + lid.x + 32u];
            let a3 = sa[base + lid.x + 48u];
            let b0 = sb[base + lid.y];
            let b1 = sb[base + lid.y + 16u];
            let b2 = sb[base + lid.y + 32u];
            let b3 = sb[base + lid.y + 48u];
            tmp[0] = tmp[0] + b0 * a0; tmp[1] = tmp[1] + b0 * a1; tmp[2] = tmp[2] + b0 * a2; tmp[3] = tmp[3] + b0 * a3;
            tmp[4] = tmp[4] + b1 * a0; tmp[5] = tmp[5] + b1 * a1; tmp[6] = tmp[6] + b1 * a2; tmp[7] = tmp[7] + b1 * a3;
            tmp[8] = tmp[8] + b2 * a0; tmp[9] = tmp[9] + b2 * a1; tmp[10] = tmp[10] + b2 * a2; tmp[11] = tmp[11] + b2 * a3;
            tmp[12] = tmp[12] + b3 * a0; tmp[13] = tmp[13] + b3 * a1; tmp[14] = tmp[14] + b3 * a2; tmp[15] = tmp[15] + b3 * a3;
        }
        for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = acc[q] + tmp[q]; }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = r0 + lid.y + 16u * j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let col = c0 + lid.x + 16u * i;
                if (col < d) {
                    let o = lr * d + col;
                    h[o] = h[o] + acc[j * 4u + i];
                }
            }
        }
    }
}
