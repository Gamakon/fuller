// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var v10: vec4<f32> = vec4<f32>(0.0);
    var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0);
    var v13: vec4<f32> = vec4<f32>(0.0);
    var v14: vec4<f32> = vec4<f32>(0.0);
    var v15: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    let g = lidx & 15u;
    let itl = lidx >> 4u;
    let ca = wid.x * 64u + g * 4u;
    let rb = wid.y * 64u + g * 4u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            var f2: f32 = 0.0;
            var f3: f32 = 0.0;
            var f4: f32 = 0.0;
            var f5: f32 = 0.0;
            var f6: f32 = 0.0;
            var f7: f32 = 0.0;
            let it = t + itl + 16u * k;
            if (it < cfg.m) {
                let base = it * cfg.d;
                if (ca + 3u < cfg.d) {
                    f0 = phi[base + ca];
                    f1 = phi[base + ca + 1u];
                    f2 = phi[base + ca + 2u];
                    f3 = phi[base + ca + 3u];
                } else {
                    if (ca < cfg.d) { f0 = phi[base + ca]; }
                    if (ca + 1u < cfg.d) { f1 = phi[base + ca + 1u]; }
                    if (ca + 2u < cfg.d) { f2 = phi[base + ca + 2u]; }
                    if (ca + 3u < cfg.d) { f3 = phi[base + ca + 3u]; }
                }
                let rbase = base + cfg.r_off + rb;
                if (rb + 3u < cfg.r_rows && cfg.r_off + rb + 3u < cfg.d) {
                    f4 = phi[rbase];
                    f5 = phi[rbase + 1u];
                    f6 = phi[rbase + 2u];
                    f7 = phi[rbase + 3u];
                } else {
                    if (rb < cfg.r_rows && cfg.r_off + rb < cfg.d) { f4 = phi[rbase]; }
                    if (rb + 1u < cfg.r_rows && cfg.r_off + rb + 1u < cfg.d) { f5 = phi[rbase + 1u]; }
                    if (rb + 2u < cfg.r_rows && cfg.r_off + rb + 2u < cfg.d) { f6 = phi[rbase + 2u]; }
                    if (rb + 3u < cfg.r_rows && cfg.r_off + rb + 3u < cfg.d) { f7 = phi[rbase + 3u]; }
                }
            }
            wg_va[(lidx + 256u * k) & 511u] = vec4<f32>(f0, f1, f2, f3);
            wg_vb[(lidx + 256u * k) & 511u] = vec4<f32>(f4, f5, f6, f7);
        }
        workgroupBarrier();
        v0 = vec4<f32>(0.0);
        v1 = vec4<f32>(0.0);
        v2 = vec4<f32>(0.0);
        v3 = vec4<f32>(0.0);
        v4 = vec4<f32>(0.0);
        v5 = vec4<f32>(0.0);
        v6 = vec4<f32>(0.0);
        v7 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            let ia = (k * 32u + lid.x) & 511u;
            let ib = (k * 32u + lid.y) & 511u;
            let ja = (k * 32u + 16u + lid.x) & 511u;
            let jb = (k * 32u + 16u + lid.y) & 511u;
            let a0 = wg_va[ia];
            let b0 = wg_vb[ib];
            let a1 = wg_va[ja];
            let b1 = wg_vb[jb];
            v0 = v0 + vec4<f32>(b0.x) * a0;
            v1 = v1 + vec4<f32>(b0.y) * a0;
            v2 = v2 + vec4<f32>(b0.z) * a0;
            v3 = v3 + vec4<f32>(b0.w) * a0;
            v4 = v4 + vec4<f32>(b1.x) * a1;
            v5 = v5 + vec4<f32>(b1.y) * a1;
            v6 = v6 + vec4<f32>(b1.z) * a1;
            v7 = v7 + vec4<f32>(b1.w) * a1;
        }
        if (((t >> 5u) & 1u) == 0u) {
            v8 = v8 + (v0 + v4);
            v9 = v9 + (v1 + v5);
            v10 = v10 + (v2 + v6);
            v11 = v11 + (v3 + v7);
        } else {
            v12 = v12 + (v0 + v4);
            v13 = v13 + (v1 + v5);
            v14 = v14 + (v2 + v6);
            v15 = v15 + (v3 + v7);
        }
        workgroupBarrier();
    }
    pv[0u] = v8 + v12;
    pv[1u] = v9 + v13;
    pv[2u] = v10 + v14;
    pv[3u] = v11 + v15;
    for (k = 0u; k < 4u; k = k + 1u) {
        let row = gid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = gid.x * 4u + k2;
                if (col < cfg.d) {
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + pv[k & 3u][k2 & 3u];
                }
            }
        }
    }
}
