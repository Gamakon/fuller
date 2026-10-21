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
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0);
    }
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        var s4 = vec4<f32>(0.0); var s5 = vec4<f32>(0.0); var s6 = vec4<f32>(0.0); var s7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let t0 = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i16 = lidx + 128u * k2;
                let it = t0 + (i16 >> 4u);
                let off = (i16 & 15u) * 4u;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                if (it < cfg.m) {
                    let rowb = it * cfg.d;
                    let c0 = cbase + off;
                    if (c0 < cfg.d) { a.x = phi[rowb + c0]; }
                    if (c0 + 1u < cfg.d) { a.y = phi[rowb + c0 + 1u]; }
                    if (c0 + 2u < cfg.d) { a.z = phi[rowb + c0 + 2u]; }
                    if (c0 + 3u < cfg.d) { a.w = phi[rowb + c0 + 3u]; }
                    let l0 = rbase + off;
                    let g0 = cfg.r_off + l0;
                    if (l0 < cfg.r_rows && g0 < cfg.d) { b.x = phi[rowb + g0]; }
                    if (l0 + 1u < cfg.r_rows && g0 + 1u < cfg.d) { b.y = phi[rowb + g0 + 1u]; }
                    if (l0 + 2u < cfg.r_rows && g0 + 2u < cfg.d) { b.z = phi[rowb + g0 + 2u]; }
                    if (l0 + 3u < cfg.r_rows && g0 + 3u < cfg.d) { b.w = phi[rowb + g0 + 3u]; }
                }
                wg_va[i16 & 511u] = a;
                wg_vb[i16 & 511u] = b;
            }
            workgroupBarrier();
            var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
            var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
            for (var k2: u32 = 0u; k2 < 16u; k2 = k2 + 1u) {
                let i20 = k2 * 32u;
                let i21 = i20 + lid.y * 2u;
                let x8 = wg_va[(i20 + lid.x) & 511u];
                let x9 = wg_va[(i20 + 16u + lid.x) & 511u];
                let y0 = wg_vb[i21 & 511u];
                let y1 = wg_vb[(i21 + 1u) & 511u];
                let y2 = wg_vb[(i21 + 16u) & 511u];
                let y3 = wg_vb[(i21 + 17u) & 511u];
                v0 = v0 + (vec4<f32>(y0.x) * x8 + vec4<f32>(y2.x) * x9);
                v1 = v1 + (vec4<f32>(y0.y) * x8 + vec4<f32>(y2.y) * x9);
                v2 = v2 + (vec4<f32>(y0.z) * x8 + vec4<f32>(y2.z) * x9);
                v3 = v3 + (vec4<f32>(y0.w) * x8 + vec4<f32>(y2.w) * x9);
                v4 = v4 + (vec4<f32>(y1.x) * x8 + vec4<f32>(y3.x) * x9);
                v5 = v5 + (vec4<f32>(y1.y) * x8 + vec4<f32>(y3.y) * x9);
                v6 = v6 + (vec4<f32>(y1.z) * x8 + vec4<f32>(y3.z) * x9);
                v7 = v7 + (vec4<f32>(y1.w) * x8 + vec4<f32>(y3.w) * x9);
            }
            s0 = s0 + v0; s1 = s1 + v1; s2 = s2 + v2; s3 = s3 + v3;
            s4 = s4 + v4; s5 = s5 + v5; s6 = s6 + v6; s7 = s7 + v7;
            workgroupBarrier();
        }
        pv[0u] = pv[0u] + s0;
        pv[1u] = pv[1u] + s1;
        pv[2u] = pv[2u] + s2;
        pv[3u] = pv[3u] + s3;
        pv[4u] = pv[4u] + s4;
        pv[5u] = pv[5u] + s5;
        pv[6u] = pv[6u] + s6;
        pv[7u] = pv[7u] + s7;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k & 15u][k2 & 3u];
                }
            }
        }
    }
}
