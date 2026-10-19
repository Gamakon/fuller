// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = idx >> 6u;
            let off = idx & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t + item < cfg.m) {
                let base = (t + item) * cfg.d;
                if (col0 + off < cfg.d) {
                    vb = phi[base + col0 + off];
                }
                if (row0 + off < cfg.r_rows && cfg.r_off + row0 + off < cfg.d) {
                    va = phi[base + cfg.r_off + row0 + off];
                }
            }
            wg_a[idx & 511u] = va;
            wg_b[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[(kk * 64u + lid.y * 4u + i) & 511u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b = wg_b[(kk * 64u + lid.x * 4u + j) & 511u];
                    acc[(i * 4u + j) & 15u] = fma(a, b, acc[(i * 4u + j) & 15u]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x * 4u + j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
