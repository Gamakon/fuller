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
    var pw: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 64u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pw[k] = vec4<f32>(0.0);
        }
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
            for (var k3: u32 = 8u; k3 < 16u; k3 = k3 + 1u) {
                pw[k3] = vec4<f32>(0.0);
            }
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let a0 = wg_va[q * 16u + lid.x];
                let a1 = wg_va[q * 16u + lid.x + 8u];
                let b = wg_vb[q * 16u + lid.y];
                pw[8] = pw[8] + b.x * a0;
                pw[9] = pw[9] + b.x * a1;
                pw[10] = pw[10] + b.y * a0;
                pw[11] = pw[11] + b.y * a1;
                pw[12] = pw[12] + b.z * a0;
                pw[13] = pw[13] + b.z * a1;
                pw[14] = pw[14] + b.w * a0;
                pw[15] = pw[15] + b.w * a1;
            }
            for (var k4: u32 = 0u; k4 < 8u; k4 = k4 + 1u) {
                pw[k4] = pw[k4] + pw[k4 + 8u];
            }
            workgroupBarrier();
        }
        for (var k5: u32 = 0u; k5 < 8u; k5 = k5 + 1u) {
            pv[k5] = pv[k5] + pw[k5];
        }
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
