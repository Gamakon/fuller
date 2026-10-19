// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    var part: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { acc[k] = 0.0; }
    let d = cfg.d;
    let m = cfg.m;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = idx >> 6u;
            let cc = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + tr < m) {
                let base = (t + tr) * d;
                if (col0 + cc < d) { fb = phi[base + col0 + cc]; }
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) { fa = phi[base + cfg.r_off + row0 + cc]; }
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) { part[k] = 0.0; }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            var pc: array<f32, 4>;
            var pd: array<f32, 4>;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                pc[q] = wg_a[k * 64u + lid.y + 16u * q];
                pd[q] = wg_b[k * 64u + lid.x + 16u * q];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    part[i * 4u + j] = part[i * 4u + j] + pc[i] * pd[j];
                }
            }
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) { acc[k] = acc[k] + part[k]; }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[i * 4u + j];
                }
            }
        }
    }
}
