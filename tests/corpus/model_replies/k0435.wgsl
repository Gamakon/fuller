// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s00 = vec4<f32>(0.0); var s01 = vec4<f32>(0.0); var s02 = vec4<f32>(0.0); var s03 = vec4<f32>(0.0);
    var s10 = vec4<f32>(0.0); var s11 = vec4<f32>(0.0); var s12 = vec4<f32>(0.0); var s13 = vec4<f32>(0.0);
    var s20 = vec4<f32>(0.0); var s21 = vec4<f32>(0.0); var s22 = vec4<f32>(0.0); var s23 = vec4<f32>(0.0);
    var s30 = vec4<f32>(0.0); var s31 = vec4<f32>(0.0); var s32 = vec4<f32>(0.0); var s33 = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0);
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    for (t = 0u; t < ((cfg.m + 31u) / 32u); t = t + 1u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t * 32u + idx / 16u;
            let q = (idx % 16u) * 4u;
            v0 = vec4<f32>(0.0);
            v1 = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    f0 = 0.0;
                    f1 = 0.0;
                    let lr = wid.y * 64u + q + k2;
                    let cc = wid.x * 64u + q + k2;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        f0 = phi[item * cfg.d + cfg.r_off + lr];
                    }
                    if (cc < cfg.d) {
                        f1 = phi[item * cfg.d + cc];
                    }
                    v0[k2] = f0;
                    v1[k2] = f1;
                }
            }
            wg_va[idx & 511u] = v0;
            wg_vb[idx & 511u] = v1;
        }
        workgroupBarrier();
        for (k = 0u; k < 32u; k = k + 4u) {
            let a0 = wg_va[(k * 16u + lid.y) & 511u];
            let b0 = wg_vb[(k * 16u + lid.x) & 511u];
            s00 = s00 + vec4<f32>(a0.x) * b0;
            s01 = s01 + vec4<f32>(a0.y) * b0;
            s02 = s02 + vec4<f32>(a0.z) * b0;
            s03 = s03 + vec4<f32>(a0.w) * b0;
            let a1 = wg_va[((k + 1u) * 16u + lid.y) & 511u];
            let b1 = wg_vb[((k + 1u) * 16u + lid.x) & 511u];
            s10 = s10 + vec4<f32>(a1.x) * b1;
            s11 = s11 + vec4<f32>(a1.y) * b1;
            s12 = s12 + vec4<f32>(a1.z) * b1;
            s13 = s13 + vec4<f32>(a1.w) * b1;
            let a2 = wg_va[((k + 2u) * 16u + lid.y) & 511u];
            let b2 = wg_vb[((k + 2u) * 16u + lid.x) & 511u];
            s20 = s20 + vec4<f32>(a2.x) * b2;
            s21 = s21 + vec4<f32>(a2.y) * b2;
            s22 = s22 + vec4<f32>(a2.z) * b2;
            s23 = s23 + vec4<f32>(a2.w) * b2;
            let a3 = wg_va[((k + 3u) * 16u + lid.y) & 511u];
            let b3 = wg_vb[((k + 3u) * 16u + lid.x) & 511u];
            s30 = s30 + vec4<f32>(a3.x) * b3;
            s31 = s31 + vec4<f32>(a3.y) * b3;
            s32 = s32 + vec4<f32>(a3.z) * b3;
            s33 = s33 + vec4<f32>(a3.w) * b3;
        }
        workgroupBarrier();
    }
    let t0 = (s00 + s10) + (s20 + s30);
    let t1 = (s01 + s11) + (s21 + s31);
    let t2 = (s02 + s12) + (s22 + s32);
    let t3 = (s03 + s13) + (s23 + s33);
    for (k = 0u; k < 4u; k = k + 1u) {
        var tot = t0;
        if (k == 1u) { tot = t1; }
        if (k == 2u) { tot = t2; }
        if (k == 3u) { tot = t3; }
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let cc = gid.x * 4u + k2;
                if (cc < cfg.d) {
                    h[lr * cfg.d + cc] = h[lr * cfg.d + cc] + tot[k2];
                }
            }
        }
    }
}
