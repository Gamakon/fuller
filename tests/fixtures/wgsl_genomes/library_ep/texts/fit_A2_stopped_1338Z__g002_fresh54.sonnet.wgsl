// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var e: u32 = 0u; e < 4u; e = e + 1u) {
            let idx = lidx + 256u * e;
            let kk = idx >> 6u;
            let cc = idx & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t + kk < m) {
                let base = (t + kk) * d;
                if (col0 + cc < d) {
                    va = phi[base + col0 + cc];
                }
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) {
                    vb = phi[base + cfg.r_off + row0 + cc];
                }
            }
            wa[idx] = va;
            wb[idx] = vb;
        }
        workgroupBarrier();
        var p: array<f32, 16>;
        for (var q: u32 = 0u; q < 16u; q = q + 1u) { p[q] = 0.0; }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                a[j] = wa[o + lid.x + 16u * j];
                b[j] = wb[o + lid.y + 16u * j];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    p[i * 4u + j] = p[i * 4u + j] + b[i] * a[j];
                }
            }
        }
        for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = acc[q] + p[q]; }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + acc[i * 4u + j];
                }
            }
        }
    }
}
