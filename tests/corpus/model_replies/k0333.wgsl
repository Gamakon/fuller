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
    var pw: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pw[k] = vec4<f32>(0.0);
        }
        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
            let tb = t + s * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i22 = lidx + 128u * k2;
                let item = tb + (i22 >> 4u);
                let cb = (i22 & 15u) * 4u;
                let col = wid.x * 64u + cb;
                let rowb = wid.y * 64u + cb;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                if (item < cfg.m) {
                    let base = item * cfg.d;
                    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                        if (col + c < cfg.d) {
                            a[c] = phi[base + col + c];
                        }
                        if (rowb + c < cfg.r_rows && cfg.r_off + rowb + c < cfg.d) {
                            b[c] = phi[base + cfg.r_off + rowb + c];
                        }
                    }
                }
                wg_va[i22 & 511u] = a;
                wg_vb[i22 & 511u] = b;
            }
            workgroupBarrier();
            var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
            var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
            for (var k2: u32 = 0u; k2 < 16u; k2 = k2 + 1u) {
                let i26 = k2 * 32u;
                let i27 = i26 + lid.y * 2u;
                let a0 = wg_va[(i26 + lid.x) & 511u];
                let a1 = wg_va[(i26 + 16u + lid.x) & 511u];
                let b0 = wg_vb[i27 & 511u];
                let b1 = wg_vb[(i27 + 1u) & 511u];
                let b2 = wg_vb[(i27 + 16u) & 511u];
                let b3 = wg_vb[(i27 + 17u) & 511u];
                v0 = v0 + (vec4<f32>(b0.x) * a0 + vec4<f32>(b2.x) * a1);
                v1 = v1 + (vec4<f32>(b0.y) * a0 + vec4<f32>(b2.y) * a1);
                v2 = v2 + (vec4<f32>(b0.z) * a0 + vec4<f32>(b2.z) * a1);
                v3 = v3 + (vec4<f32>(b0.w) * a0 + vec4<f32>(b2.w) * a1);
                v4 = v4 + (vec4<f32>(b1.x) * a0 + vec4<f32>(b3.x) * a1);
                v5 = v5 + (vec4<f32>(b1.y) * a0 + vec4<f32>(b3.y) * a1);
                v6 = v6 + (vec4<f32>(b1.z) * a0 + vec4<f32>(b3.z) * a1);
                v7 = v7 + (vec4<f32>(b1.w) * a0 + vec4<f32>(b3.w) * a1);
            }
            pw[0] = pw[0] + v0; pw[1] = pw[1] + v1; pw[2] = pw[2] + v2; pw[3] = pw[3] + v3;
            pw[4] = pw[4] + v4; pw[5] = pw[5] + v5; pw[6] = pw[6] + v6; pw[7] = pw[7] + v7;
            workgroupBarrier();
        }
        let o = ((t >> 7u) & 1u) * 8u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pv[o + k] = pv[o + k] + pw[k];
        }
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let tot = pv[k] + pv[k + 8u];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + tot[k2];
                }
            }
        }
    }
}
