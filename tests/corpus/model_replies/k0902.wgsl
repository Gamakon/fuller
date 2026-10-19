// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let lc = lid.x * 4u;
    let lr = lid.y * 4u;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        workgroupBarrier();
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let i = lidx + q * 256u;
            let it = i >> 6u;
            let cc = i & 63u;
            let t = t0 + it;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (colbase + cc < d) {
                    vb = phi[base + colbase + cc];
                }
                let rl = rowbase + cc;
                if (rl < r_rows && r_off + rl < d) {
                    va = phi[base + r_off + rl];
                }
            }
            wg_a[i & 1023u] = va;
            wg_b[i & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let b = vec4<f32>(wg_b[(o + lc) & 1023u], wg_b[(o + lc + 1u) & 1023u], wg_b[(o + lc + 2u) & 1023u], wg_b[(o + lc + 3u) & 1023u]);
            let r0 = wg_a[(o + lr) & 1023u];
            let r1 = wg_a[(o + lr + 1u) & 1023u];
            let r2 = wg_a[(o + lr + 2u) & 1023u];
            let r3 = wg_a[(o + lr + 3u) & 1023u];
            a0 = fma(vec4<f32>(r0), b, a0);
            a1 = fma(vec4<f32>(r1), b, a1);
            a2 = fma(vec4<f32>(r2), b, a2);
            a3 = fma(vec4<f32>(r3), b, a3);
        }
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let rl = rowbase + lr + j;
        if (rl < r_rows && r_off + rl < d) {
            var v = a0;
            if (j == 1u) { v = a1; }
            if (j == 2u) { v = a2; }
            if (j == 3u) { v = a3; }
            let rb = rl * d;
            let c0 = colbase + lc;
            if (c0 < d) { h[rb + c0] = h[rb + c0] + v.x; }
            if (c0 + 1u < d) { h[rb + c0 + 1u] = h[rb + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[rb + c0 + 2u] = h[rb + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[rb + c0 + 3u] = h[rb + c0 + 3u] + v.w; }
        }
    }
}
