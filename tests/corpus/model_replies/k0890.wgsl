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
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = idx >> 6u;
            let col = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + item < m) {
                let base = (t + item) * d;
                let c = wid.x * 64u + col;
                if (c < d) {
                    fb = phi[base + c];
                }
                let lr = wid.y * 64u + col;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    fa = phi[base + gr];
                }
            }
            wg_a[idx & 511u] = fa;
            wg_b[idx & 511u] = fb;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let ra = s * 64u + lid.y * 4u;
            let rb = s * 64u + lid.x * 4u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let av = wg_a[(ra + i) & 511u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = fma(av, wg_b[(rb + j) & 511u], acc[q]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x * 4u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
