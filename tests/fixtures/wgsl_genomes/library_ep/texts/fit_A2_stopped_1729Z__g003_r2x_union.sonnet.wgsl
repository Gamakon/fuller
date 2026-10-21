// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
    var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
    var v8 = vec4<f32>(0.0); var v9 = vec4<f32>(0.0); var v10 = vec4<f32>(0.0); var v11 = vec4<f32>(0.0);
    var v12 = vec4<f32>(0.0); var v13 = vec4<f32>(0.0);
    var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0); var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
    var q4 = vec4<f32>(0.0); var q5 = vec4<f32>(0.0); var q6 = vec4<f32>(0.0); var q7 = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var j: u32 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let tb = t + (k * 32u);
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i13 = lidx + (128u * k2);
                let item = tb + (i13 >> 4u);
                let cc = (i13 & 15u) * 4u;
                var fa = vec4<f32>(0.0);
                var fb = vec4<f32>(0.0);
                if (item < cfg.m) {
                    let base = item * cfg.d;
                    for (j = 0u; j < 4u; j = j + 1u) {
                        let col = (wid.x * 64u) + cc + j;
                        if (col < cfg.d) {
                            fa[j] = phi[base + col];
                        }
                        let rl = (wid.y * 64u) + cc + j;
                        if (rl < cfg.r_rows && (cfg.r_off + rl) < cfg.d) {
                            fb[j] = phi[base + cfg.r_off + rl];
                        }
                    }
                }
                wg_va[i13 & 511u] = fa;
                wg_vb[i13 & 511u] = fb;
            }
            workgroupBarrier();
            q0 = vec4<f32>(0.0); q1 = vec4<f32>(0.0); q2 = vec4<f32>(0.0); q3 = vec4<f32>(0.0);
            q4 = vec4<f32>(0.0); q5 = vec4<f32>(0.0); q6 = vec4<f32>(0.0); q7 = vec4<f32>(0.0);
            for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
                let i17 = k2 * 32u;
                let i18 = i17 + (lid.y * 2u);
                v8 = wg_va[(i17 + lid.x) & 511u];
                v9 = wg_va[(i17 + 16u + lid.x) & 511u];
                v10 = wg_vb[i18 & 511u];
                v11 = wg_vb[(i18 + 1u) & 511u];
                v12 = wg_vb[(i18 + 16u) & 511u];
                v13 = wg_vb[(i18 + 17u) & 511u];
                q0 = q0 + ((vec4<f32>(v10.x) * v8) + (vec4<f32>(v12.x) * v9));
                q1 = q1 + ((vec4<f32>(v10.y) * v8) + (vec4<f32>(v12.y) * v9));
                q2 = q2 + ((vec4<f32>(v10.z) * v8) + (vec4<f32>(v12.z) * v9));
                q3 = q3 + ((vec4<f32>(v10.w) * v8) + (vec4<f32>(v12.w) * v9));
                q4 = q4 + ((vec4<f32>(v11.x) * v8) + (vec4<f32>(v13.x) * v9));
                q5 = q5 + ((vec4<f32>(v11.y) * v8) + (vec4<f32>(v13.y) * v9));
                q6 = q6 + ((vec4<f32>(v11.z) * v8) + (vec4<f32>(v13.z) * v9));
                q7 = q7 + ((vec4<f32>(v11.w) * v8) + (vec4<f32>(v13.w) * v9));
            }
            a0 = a0 + q0; a1 = a1 + q1; a2 = a2 + q2; a3 = a3 + q3;
            a4 = a4 + q4; a5 = a5 + q5; a6 = a6 + q6; a7 = a7 + q7;
            workgroupBarrier();
        }
        pv[0u] = pv[0u] + a0; pv[1u] = pv[1u] + a1; pv[2u] = pv[2u] + a2; pv[3u] = pv[3u] + a3;
        pv[4u] = pv[4u] + a4; pv[5u] = pv[5u] + a5; pv[6u] = pv[6u] + a6; pv[7u] = pv[7u] + a7;
        a0 = vec4<f32>(0.0); a1 = vec4<f32>(0.0); a2 = vec4<f32>(0.0); a3 = vec4<f32>(0.0);
        a4 = vec4<f32>(0.0); a5 = vec4<f32>(0.0); a6 = vec4<f32>(0.0); a7 = vec4<f32>(0.0);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        let row = (gid.y * 8u) + k;
        if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = (gid.x * 4u) + k2;
                if (col < cfg.d) {
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + pv[k][k2];
                }
            }
        }
    }
}
