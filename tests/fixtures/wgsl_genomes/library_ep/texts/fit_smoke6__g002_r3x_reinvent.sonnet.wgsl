// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc0: array<vec4<f32>, 4>;
    var acc1: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc0[i] = vec4<f32>(0.0);
        acc1[i] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let row_base = wid.y * 64u;
    let col_base = wid.x * 64u;
    let ra = lid.y * 4u;
    let cb = lid.x * 4u;
    for (var t = 0u; t < m; t = t + 16u) {
        for (var q = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let it = idx >> 6u;
            let cc = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + it < m) {
                let lr = row_base + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    fa = phi[(t + it) * d + cfg.r_off + lr];
                }
                if (col_base + cc < d) {
                    fb = phi[(t + it) * d + col_base + cc];
                }
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 2u) {
            let o0 = k * 64u;
            let o1 = o0 + 64u;
            let b0 = vec4<f32>(wg_b[o0 + cb], wg_b[o0 + cb + 1u], wg_b[o0 + cb + 2u], wg_b[o0 + cb + 3u]);
            let b1 = vec4<f32>(wg_b[o1 + cb], wg_b[o1 + cb + 1u], wg_b[o1 + cb + 2u], wg_b[o1 + cb + 3u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                acc0[i] = acc0[i] + wg_a[o0 + ra + i] * b0;
                acc1[i] = acc1[i] + wg_a[o1 + ra + i] * b1;
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row_base + ra + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let s = acc0[i] + acc1[i];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col_base + cb + j;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + s[j];
                }
            }
        }
    }
}
