// grid 16 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0);
    }
    var cs: u32 = 16u;
    for (var k: u32 = 0u; k < 10u; k = k + 1u) {
        if (cs * cs < cfg.m) { cs = cs * 2u; }
    }
    let colBase = wid.x * 128u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let e = lidx + 128u * k2;
            let it = t + (e >> 5u);
            let c = colBase + (e & 31u) * 4u;
            var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
            if (it < cfg.m) {
                let base = it * cfg.d + c;
                if (c < cfg.d) { f0 = phi[base]; }
                if (c + 1u < cfg.d) { f1 = phi[base + 1u]; }
                if (c + 2u < cfg.d) { f2 = phi[base + 2u]; }
                if (c + 3u < cfg.d) { f3 = phi[base + 3u]; }
            }
            wg_va[e & 511u] = vec4<f32>(f0, f1, f2, f3);
        }
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let e = lidx + 128u * k2;
            let it = t + (e >> 4u);
            let lr = rowBase + (e & 15u) * 4u;
            let gr = cfg.r_off + lr;
            var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
            if (it < cfg.m) {
                let base = it * cfg.d + gr;
                if (lr < cfg.r_rows && gr < cfg.d) { f4 = phi[base]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < cfg.d) { f5 = phi[base + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < cfg.d) { f6 = phi[base + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < cfg.d) { f7 = phi[base + 3u]; }
            }
            wg_vb[e & 511u] = vec4<f32>(f4, f5, f6, f7);
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let a0 = wg_va[(j * 32u + lid.x) & 511u];
            let a1 = wg_va[(j * 32u + 16u + lid.x) & 511u];
            let b0 = wg_vb[(j * 16u + lid.y) & 511u];
            let b1 = wg_vb[(j * 16u + 8u + lid.y) & 511u];
            v0 = v0 + vec4<f32>(b0.x) * a0;
            v1 = v1 + vec4<f32>(b0.y) * a0;
            v2 = v2 + vec4<f32>(b0.z) * a0;
            v3 = v3 + vec4<f32>(b0.w) * a0;
            v4 = v4 + vec4<f32>(b0.x) * a1;
            v5 = v5 + vec4<f32>(b0.y) * a1;
            v6 = v6 + vec4<f32>(b0.z) * a1;
            v7 = v7 + vec4<f32>(b0.w) * a1;
            v8 = v8 + vec4<f32>(b1.x) * a0;
            v9 = v9 + vec4<f32>(b1.y) * a0;
            v10 = v10 + vec4<f32>(b1.z) * a0;
            v11 = v11 + vec4<f32>(b1.w) * a0;
            v12 = v12 + vec4<f32>(b1.x) * a1;
            v13 = v13 + vec4<f32>(b1.y) * a1;
            v14 = v14 + vec4<f32>(b1.z) * a1;
            v15 = v15 + vec4<f32>(b1.w) * a1;
        }
        workgroupBarrier();
        if ((((t + 16u) & (cs - 1u)) == 0u) || (t + 16u >= cfg.m)) {
            pv[0u] = pv[0u] + v0; v0 = vec4<f32>(0.0);
            pv[1u] = pv[1u] + v1; v1 = vec4<f32>(0.0);
            pv[2u] = pv[2u] + v2; v2 = vec4<f32>(0.0);
            pv[3u] = pv[3u] + v3; v3 = vec4<f32>(0.0);
            pv[4u] = pv[4u] + v4; v4 = vec4<f32>(0.0);
            pv[5u] = pv[5u] + v5; v5 = vec4<f32>(0.0);
            pv[6u] = pv[6u] + v6; v6 = vec4<f32>(0.0);
            pv[7u] = pv[7u] + v7; v7 = vec4<f32>(0.0);
            pv[8u] = pv[8u] + v8; v8 = vec4<f32>(0.0);
            pv[9u] = pv[9u] + v9; v9 = vec4<f32>(0.0);
            pv[10u] = pv[10u] + v10; v10 = vec4<f32>(0.0);
            pv[11u] = pv[11u] + v11; v11 = vec4<f32>(0.0);
            pv[12u] = pv[12u] + v12; v12 = vec4<f32>(0.0);
            pv[13u] = pv[13u] + v13; v13 = vec4<f32>(0.0);
            pv[14u] = pv[14u] + v14; v14 = vec4<f32>(0.0);
            pv[15u] = pv[15u] + v15; v15 = vec4<f32>(0.0);
        }
    }
    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let lr = rowBase + g * 32u + lid.y * 4u + q;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                for (var cg: u32 = 0u; cg < 2u; cg = cg + 1u) {
                    let acc = pv[(g * 8u + cg * 4u + q) & 15u];
                    for (var q2: u32 = 0u; q2 < 4u; q2 = q2 + 1u) {
                        let c = colBase + cg * 64u + lid.x * 4u + q2;
                        if (c < cfg.d) {
                            let idx = lr * cfg.d + c;
                            h[idx] = h[idx] + acc[q2 & 3u];
                        }
                    }
                }
            }
        }
    }
}
