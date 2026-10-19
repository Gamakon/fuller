// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    let col0 = wid.x * 128u;
    let row0 = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let idx = lidx + 256u * k2;
            let item = t + (idx >> 5u);
            let col = col0 + (idx & 31u) * 4u;
            var a = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d + col;
                if (col + 3u < cfg.d) {
                    a = vec4<f32>(phi[base], phi[base + 1u], phi[base + 2u], phi[base + 3u]);
                } else {
                    if (col < cfg.d) { a.x = phi[base]; }
                    if (col + 1u < cfg.d) { a.y = phi[base + 1u]; }
                    if (col + 2u < cfg.d) { a.z = phi[base + 2u]; }
                }
            }
            wg_va[idx & 511u] = a;
        }
        {
            let idx = lidx;
            let item = t + (idx >> 4u);
            let lr = row0 + (idx & 15u) * 4u;
            let gr = cfg.r_off + lr;
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d + gr;
                if (lr < cfg.r_rows && gr < cfg.d) { b.x = phi[base]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < cfg.d) { b.y = phi[base + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < cfg.d) { b.z = phi[base + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < cfg.d) { b.w = phi[base + 3u]; }
            }
            wg_vb[idx & 511u] = b;
        }
        workgroupBarrier();
        v8 = vec4<f32>(0.0); v9 = vec4<f32>(0.0); v10 = vec4<f32>(0.0); v11 = vec4<f32>(0.0);
        v12 = vec4<f32>(0.0); v13 = vec4<f32>(0.0); v14 = vec4<f32>(0.0); v15 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            v16 = wg_va[(k * 32u + lid.x) & 511u];
            v17 = wg_vb[(k * 16u + lid.y * 2u) & 511u];
            v18 = wg_vb[(k * 16u + lid.y * 2u + 1u) & 511u];
            v8 = v8 + vec4<f32>(v17[0u]) * v16;
            v9 = v9 + vec4<f32>(v17[1u]) * v16;
            v10 = v10 + vec4<f32>(v17[2u]) * v16;
            v11 = v11 + vec4<f32>(v17[3u]) * v16;
            v12 = v12 + vec4<f32>(v18[0u]) * v16;
            v13 = v13 + vec4<f32>(v18[1u]) * v16;
            v14 = v14 + vec4<f32>(v18[2u]) * v16;
            v15 = v15 + vec4<f32>(v18[3u]) * v16;
        }
        v0 = v0 + v8; v1 = v1 + v9; v2 = v2 + v10; v3 = v3 + v11;
        v4 = v4 + v12; v5 = v5 + v13; v6 = v6 + v14; v7 = v7 + v15;
        workgroupBarrier();
    }
    pv[0u] = v0; pv[1u] = v1; pv[2u] = v2; pv[3u] = v3;
    pv[4u] = v4; pv[5u] = v5; pv[6u] = v6; pv[7u] = v7;
    for (k = 0u; k < 8u; k = k + 1u) {
        let lr = row0 + lid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let acc = pv[k & 15u];
            let cb = col0 + lid.x * 4u;
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (cb + k2 < cfg.d) {
                    let hi = lr * cfg.d + cb + k2;
                    h[hi] = h[hi] + acc[k2 & 3u];
                }
            }
        }
    }
}
