// grid 16 8 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
    var c4 = vec4<f32>(0.0); var c5 = vec4<f32>(0.0); var c6 = vec4<f32>(0.0); var c7 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 128u;
    let rowbase = wid.y * 32u;
    for (var t: u32 = 0u; t < m; t = t + 64u) {
        var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
        var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
            let tb = t + s * 16u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let idx = lidx + 128u * q;
                let item = tb + (idx >> 5u);
                let col = colbase + (idx & 31u) * 4u;
                var a = vec4<f32>(0.0);
                if (item < m) {
                    let p = item * d + col;
                    if (col < d) { a.x = phi[p]; }
                    if (col + 1u < d) { a.y = phi[p + 1u]; }
                    if (col + 2u < d) { a.z = phi[p + 2u]; }
                    if (col + 3u < d) { a.w = phi[p + 3u]; }
                }
                wg_va[idx & 511u] = a;
            }
            {
                let item = tb + (lidx >> 3u);
                let lr = rowbase + (lidx & 7u) * 4u;
                let gr = cfg.r_off + lr;
                var b = vec4<f32>(0.0);
                if (item < m) {
                    let p = item * d + gr;
                    if (lr < cfg.r_rows && gr < d) { b.x = phi[p]; }
                    if (lr + 1u < cfg.r_rows && gr + 1u < d) { b.y = phi[p + 1u]; }
                    if (lr + 2u < cfg.r_rows && gr + 2u < d) { b.z = phi[p + 2u]; }
                    if (lr + 3u < cfg.r_rows && gr + 3u < d) { b.w = phi[p + 3u]; }
                }
                wg_vb[lidx & 511u] = b;
            }
            workgroupBarrier();
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                let a0 = wg_va[(k * 32u + lid.x * 2u) & 511u];
                let a1 = wg_va[(k * 32u + lid.x * 2u + 1u) & 511u];
                let b = wg_vb[(k * 8u + lid.y) & 511u];
                v0 = v0 + vec4<f32>(b.x) * a0;
                v1 = v1 + vec4<f32>(b.x) * a1;
                v2 = v2 + vec4<f32>(b.y) * a0;
                v3 = v3 + vec4<f32>(b.y) * a1;
                v4 = v4 + vec4<f32>(b.z) * a0;
                v5 = v5 + vec4<f32>(b.z) * a1;
                v6 = v6 + vec4<f32>(b.w) * a0;
                v7 = v7 + vec4<f32>(b.w) * a1;
            }
            workgroupBarrier();
        }
        c0 = c0 + v0; c1 = c1 + v1; c2 = c2 + v2; c3 = c3 + v3;
        c4 = c4 + v4; c5 = c5 + v5; c6 = c6 + v6; c7 = c7 + v7;
    }
    var pv: array<vec4<f32>, 8>;
    pv[0u] = c0; pv[1u] = c1; pv[2u] = c2; pv[3u] = c3;
    pv[4u] = c4; pv[5u] = c5; pv[6u] = c6; pv[7u] = c7;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var c: u32 = 0u; c < 8u; c = c + 1u) {
                let col = colbase + lid.x * 8u + c;
                if (col < d) {
                    let hi = lr * d + col;
                    h[hi] = h[hi] + pv[(i * 2u + (c >> 2u)) & 7u][c & 3u];
                }
            }
        }
    }
}
