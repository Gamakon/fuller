// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pe: array<vec4<f32>, 8>;
    var po: array<vec4<f32>, 8>;
    var tm: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pe[k] = vec4<f32>(0.0);
        po[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let idx = lidx + 128u * k2;
            let item = t + (idx >> 4u);
            let c4 = (idx & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                let cx = wid.x * 64u + c4;
                let ry = wid.y * 64u + c4;
                if (cx < cfg.d) { va.x = phi[base + cx]; }
                if (cx + 1u < cfg.d) { va.y = phi[base + cx + 1u]; }
                if (cx + 2u < cfg.d) { va.z = phi[base + cx + 2u]; }
                if (cx + 3u < cfg.d) { va.w = phi[base + cx + 3u]; }
                if (ry < cfg.r_rows && cfg.r_off + ry < cfg.d) { vb.x = phi[base + cfg.r_off + ry]; }
                if (ry + 1u < cfg.r_rows && cfg.r_off + ry + 1u < cfg.d) { vb.y = phi[base + cfg.r_off + ry + 1u]; }
                if (ry + 2u < cfg.r_rows && cfg.r_off + ry + 2u < cfg.d) { vb.z = phi[base + cfg.r_off + ry + 2u]; }
                if (ry + 3u < cfg.r_rows && cfg.r_off + ry + 3u < cfg.d) { vb.w = phi[base + cfg.r_off + ry + 3u]; }
            }
            wg_va[idx] = va;
            wg_vb[idx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            tm[k] = vec4<f32>(0.0);
        }
        for (var k2: u32 = 0u; k2 < 16u; k2 = k2 + 1u) {
            let i17 = k2 * 32u;
            let i18 = i17 + lid.y * 2u;
            let a0 = wg_va[i17 + lid.x];
            let a1 = wg_va[i17 + 16u + lid.x];
            let b0 = wg_vb[i18];
            let b1 = wg_vb[i18 + 1u];
            let b2 = wg_vb[i18 + 16u];
            let b3 = wg_vb[i18 + 17u];
            tm[0] = tm[0] + (vec4<f32>(b0.x) * a0 + vec4<f32>(b2.x) * a1);
            tm[1] = tm[1] + (vec4<f32>(b0.y) * a0 + vec4<f32>(b2.y) * a1);
            tm[2] = tm[2] + (vec4<f32>(b0.z) * a0 + vec4<f32>(b2.z) * a1);
            tm[3] = tm[3] + (vec4<f32>(b0.w) * a0 + vec4<f32>(b2.w) * a1);
            tm[4] = tm[4] + (vec4<f32>(b1.x) * a0 + vec4<f32>(b3.x) * a1);
            tm[5] = tm[5] + (vec4<f32>(b1.y) * a0 + vec4<f32>(b3.y) * a1);
            tm[6] = tm[6] + (vec4<f32>(b1.z) * a0 + vec4<f32>(b3.z) * a1);
            tm[7] = tm[7] + (vec4<f32>(b1.w) * a0 + vec4<f32>(b3.w) * a1);
        }
        if (((t >> 5u) & 1u) == 0u) {
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                pe[k] = pe[k] + tm[k];
            }
        } else {
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                po[k] = po[k] + tm[k];
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let s = pe[k] + po[k];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + s[k2];
                }
            }
        }
    }
}
