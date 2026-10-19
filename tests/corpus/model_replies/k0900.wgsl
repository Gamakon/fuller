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
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = t + (idx >> 6u);
            let cc = idx & 63u;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (it < cfg.m) {
                if (colBase + cc < d) {
                    fb = phi[it * d + colBase + cc];
                }
                if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < d) {
                    fa = phi[it * d + cfg.r_off + rowBase + cc];
                }
            }
            wg_a[idx & 511u] = fa;
            wg_b[idx & 511u] = fb;
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            var av: array<f32, 4>;
            var bv: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                av[i & 3u] = wg_a[(j * 64u + lid.y * 4u + i) & 511u];
                bv[i & 3u] = wg_b[(j * 64u + lid.x * 4u + i) & 511u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                    let q = (i * 4u + c) & 15u;
                    acc[q] = fma(av[i & 3u], bv[c & 3u], acc[q]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                let col = colBase + lid.x * 4u + c;
                if (col < d) {
                    let o = lr * d + col;
                    h[o] = h[o] + acc[(i * 4u + c) & 15u];
                }
            }
        }
    }
}
