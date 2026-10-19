// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
    var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
    var e0 = vec4<f32>(0.0); var e1 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0); var e3 = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let i6 = lidx + 256u * k;
            let i7 = t + (i6 >> 4u);
            let i8 = wid.x * 64u + (i6 & 15u) * 4u;
            let i9 = wid.y * 64u + (i6 & 15u) * 4u;
            var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
            var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
            if (i7 < cfg.m) {
                let i10 = i7 * cfg.d;
                if (i8 < cfg.d) { f0 = phi[i10 + i8]; }
                if (i8 + 1u < cfg.d) { f1 = phi[i10 + i8 + 1u]; }
                if (i8 + 2u < cfg.d) { f2 = phi[i10 + i8 + 2u]; }
                if (i8 + 3u < cfg.d) { f3 = phi[i10 + i8 + 3u]; }
                if (i9 < cfg.r_rows && cfg.r_off + i9 < cfg.d) { f4 = phi[i10 + cfg.r_off + i9]; }
                if (i9 + 1u < cfg.r_rows && cfg.r_off + i9 + 1u < cfg.d) { f5 = phi[i10 + cfg.r_off + i9 + 1u]; }
                if (i9 + 2u < cfg.r_rows && cfg.r_off + i9 + 2u < cfg.d) { f6 = phi[i10 + cfg.r_off + i9 + 2u]; }
                if (i9 + 3u < cfg.r_rows && cfg.r_off + i9 + 3u < cfg.d) { f7 = phi[i10 + cfg.r_off + i9 + 3u]; }
            }
            wg_va[i6 & 511u] = vec4<f32>(f0, f1, f2, f3);
            wg_vb[i6 & 511u] = vec4<f32>(f4, f5, f6, f7);
        }
        workgroupBarrier();
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        var u0 = vec4<f32>(0.0); var u1 = vec4<f32>(0.0); var u2 = vec4<f32>(0.0); var u3 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            let x0 = wg_va[(2u * k) * 16u + lid.x];
            let y0 = wg_vb[(2u * k) * 16u + lid.y];
            let x1 = wg_va[(2u * k + 1u) * 16u + lid.x];
            let y1 = wg_vb[(2u * k + 1u) * 16u + lid.y];
            s0 = s0 + vec4<f32>(y0.x) * x0;
            s1 = s1 + vec4<f32>(y0.y) * x0;
            s2 = s2 + vec4<f32>(y0.z) * x0;
            s3 = s3 + vec4<f32>(y0.w) * x0;
            u0 = u0 + vec4<f32>(y1.x) * x1;
            u1 = u1 + vec4<f32>(y1.y) * x1;
            u2 = u2 + vec4<f32>(y1.z) * x1;
            u3 = u3 + vec4<f32>(y1.w) * x1;
        }
        let sel = (t >> 5u) & 3u;
        if (sel == 0u) {
            a0 = a0 + (s0 + u0); a1 = a1 + (s1 + u1); a2 = a2 + (s2 + u2); a3 = a3 + (s3 + u3);
        } else if (sel == 1u) {
            b0 = b0 + (s0 + u0); b1 = b1 + (s1 + u1); b2 = b2 + (s2 + u2); b3 = b3 + (s3 + u3);
        } else if (sel == 2u) {
            c0 = c0 + (s0 + u0); c1 = c1 + (s1 + u1); c2 = c2 + (s2 + u2); c3 = c3 + (s3 + u3);
        } else {
            e0 = e0 + (s0 + u0); e1 = e1 + (s1 + u1); e2 = e2 + (s2 + u2); e3 = e3 + (s3 + u3);
        }
        workgroupBarrier();
    }
    pv[0u] = (a0 + b0) + (c0 + e0);
    pv[1u] = (a1 + b1) + (c1 + e1);
    pv[2u] = (a2 + b2) + (c2 + e2);
    pv[3u] = (a3 + b3) + (c3 + e3);
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pv[k][k2];
                }
            }
        }
    }
}
