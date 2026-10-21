// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 64u) {
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        var s4 = vec4<f32>(0.0); var s5 = vec4<f32>(0.0); var s6 = vec4<f32>(0.0); var s7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i5 = t + k * 16u;
            for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
                let i7 = lidx + 128u * k2;
                let i8 = i5 + (i7 >> 4u);
                let i9 = (i7 & 15u) * 4u;
                var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
                var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
                if (i8 < cfg.m) {
                    let cb = wid.x * 64u + i9;
                    let rb = wid.y * 64u + i9;
                    let i10 = i8 * cfg.d + cb;
                    if (cb < cfg.d) { f0 = phi[i10]; }
                    if (cb + 1u < cfg.d) { f1 = phi[i10 + 1u]; }
                    if (cb + 2u < cfg.d) { f2 = phi[i10 + 2u]; }
                    if (cb + 3u < cfg.d) { f3 = phi[i10 + 3u]; }
                    let j10 = i8 * cfg.d + cfg.r_off + rb;
                    if (rb < cfg.r_rows && cfg.r_off + rb < cfg.d) { f4 = phi[j10]; }
                    if (rb + 1u < cfg.r_rows && cfg.r_off + rb + 1u < cfg.d) { f5 = phi[j10 + 1u]; }
                    if (rb + 2u < cfg.r_rows && cfg.r_off + rb + 2u < cfg.d) { f6 = phi[j10 + 2u]; }
                    if (rb + 3u < cfg.r_rows && cfg.r_off + rb + 3u < cfg.d) { f7 = phi[j10 + 3u]; }
                }
                wg_va[i7] = vec4<f32>(f0, f1, f2, f3);
                wg_vb[i7] = vec4<f32>(f4, f5, f6, f7);
            }
            workgroupBarrier();
            var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
            var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let a0 = wg_va[q * 16u + lid.x];
                let a1 = wg_va[q * 16u + lid.x + 8u];
                let b = wg_vb[q * 16u + lid.y];
                v0 = v0 + vec4<f32>(b.x) * a0;
                v1 = v1 + vec4<f32>(b.x) * a1;
                v2 = v2 + vec4<f32>(b.y) * a0;
                v3 = v3 + vec4<f32>(b.y) * a1;
                v4 = v4 + vec4<f32>(b.z) * a0;
                v5 = v5 + vec4<f32>(b.z) * a1;
                v6 = v6 + vec4<f32>(b.w) * a0;
                v7 = v7 + vec4<f32>(b.w) * a1;
            }
            s0 = s0 + v0; s1 = s1 + v1; s2 = s2 + v2; s3 = s3 + v3;
            s4 = s4 + v4; s5 = s5 + v5; s6 = s6 + v6; s7 = s7 + v7;
            workgroupBarrier();
        }
        pv[0] = pv[0] + s0; pv[1] = pv[1] + s1; pv[2] = pv[2] + s2; pv[3] = pv[3] + s3;
        pv[4] = pv[4] + s4; pv[5] = pv[5] + s5; pv[6] = pv[6] + s6; pv[7] = pv[7] + s7;
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
                let c0 = wid.x * 64u + hf * 32u + lid.x * 4u;
                let acc = pv[k * 2u + hf];
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    if (c0 + e < cfg.d) {
                        let idx = lr * cfg.d + c0 + e;
                        h[idx] = h[idx] + acc[e];
                    }
                }
            }
        }
    }
}
