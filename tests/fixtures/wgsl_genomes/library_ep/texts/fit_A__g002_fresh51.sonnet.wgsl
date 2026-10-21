// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 2048>;
var<workgroup> wb: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }
    for (var t0 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var k = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + tr;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let base = t * d;
                if (col0 + cc < d) { va = phi[base + col0 + cc]; }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) { vb = phi[base + cfg.r_off + lr]; }
            }
            wa[idx] = va;
            wb[idx] = vb;
        }
        workgroupBarrier();
        var p: array<f32, 16>;
        for (var q = 0u; q < 16u; q = q + 1u) { p[q] = 0.0; }
        for (var k = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let a0 = wa[o + lid.x];
            let a1 = wa[o + lid.x + 16u];
            let a2 = wa[o + lid.x + 32u];
            let a3 = wa[o + lid.x + 48u];
            let b0 = wb[o + lid.y];
            let b1 = wb[o + lid.y + 16u];
            let b2 = wb[o + lid.y + 32u];
            let b3 = wb[o + lid.y + 48u];
            p[0] = p[0] + b0 * a0; p[1] = p[1] + b0 * a1; p[2] = p[2] + b0 * a2; p[3] = p[3] + b0 * a3;
            p[4] = p[4] + b1 * a0; p[5] = p[5] + b1 * a1; p[6] = p[6] + b1 * a2; p[7] = p[7] + b1 * a3;
            p[8] = p[8] + b2 * a0; p[9] = p[9] + b2 * a1; p[10] = p[10] + b2 * a2; p[11] = p[11] + b2 * a3;
            p[12] = p[12] + b3 * a0; p[13] = p[13] + b3 * a1; p[14] = p[14] + b3 * a2; p[15] = p[15] + b3 * a3;
        }
        for (var q = 0u; q < 16u; q = q + 1u) { acc[q] = acc[q] + p[q]; }
        workgroupBarrier();
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y + 16u * j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = col0 + lid.x + 16u * i;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + acc[j * 4u + i];
                }
            }
        }
    }
}
