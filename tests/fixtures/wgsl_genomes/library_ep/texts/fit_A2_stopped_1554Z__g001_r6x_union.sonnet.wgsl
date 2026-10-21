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
    var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
    var g0 = vec4<f32>(0.0); var g1 = vec4<f32>(0.0); var g2 = vec4<f32>(0.0); var g3 = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i4 = lidx + 256u * k;
            let item = t + (i4 >> 4u);
            let cx = wid.x * 64u + (i4 & 15u) * 4u;
            let ry = wid.y * 64u + (i4 & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                if (cx < cfg.d) { va.x = phi[base + cx]; }
                if (cx + 1u < cfg.d) { va.y = phi[base + cx + 1u]; }
                if (cx + 2u < cfg.d) { va.z = phi[base + cx + 2u]; }
                if (cx + 3u < cfg.d) { va.w = phi[base + cx + 3u]; }
                if (ry < cfg.r_rows && cfg.r_off + ry < cfg.d) { vb.x = phi[base + cfg.r_off + ry]; }
                if (ry + 1u < cfg.r_rows && cfg.r_off + ry + 1u < cfg.d) { vb.y = phi[base + cfg.r_off + ry + 1u]; }
                if (ry + 2u < cfg.r_rows && cfg.r_off + ry + 2u < cfg.d) { vb.z = phi[base + cfg.r_off + ry + 2u]; }
                if (ry + 3u < cfg.r_rows && cfg.r_off + ry + 3u < cfg.d) { vb.w = phi[base + cfg.r_off + ry + 3u]; }
            }
            wg_va[i4 & 511u] = va;
            wg_vb[i4 & 511u] = vb;
        }
        workgroupBarrier();
        var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
        var e0 = vec4<f32>(0.0); var e1 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0); var e3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let pa = wg_va[(((2u * k) * 16u) + lid.x) & 511u];
            let pb = wg_vb[(((2u * k) * 16u) + lid.y) & 511u];
            c0 = c0 + vec4<f32>(pb.x) * pa;
            c1 = c1 + vec4<f32>(pb.y) * pa;
            c2 = c2 + vec4<f32>(pb.z) * pa;
            c3 = c3 + vec4<f32>(pb.w) * pa;
            let qa = wg_va[(((2u * k + 1u) * 16u) + lid.x) & 511u];
            let qb = wg_vb[(((2u * k + 1u) * 16u) + lid.y) & 511u];
            e0 = e0 + vec4<f32>(qb.x) * qa;
            e1 = e1 + vec4<f32>(qb.y) * qa;
            e2 = e2 + vec4<f32>(qb.z) * qa;
            e3 = e3 + vec4<f32>(qb.w) * qa;
        }
        a0 = a0 + (c0 + e0);
        a1 = a1 + (c1 + e1);
        a2 = a2 + (c2 + e2);
        a3 = a3 + (c3 + e3);
        if (((t >> 5u) & 7u) == 7u) {
            g0 = g0 + a0; g1 = g1 + a1; g2 = g2 + a2; g3 = g3 + a3;
            a0 = vec4<f32>(0.0); a1 = vec4<f32>(0.0); a2 = vec4<f32>(0.0); a3 = vec4<f32>(0.0);
        }
        workgroupBarrier();
    }
    pv[0u] = g0 + a0;
    pv[1u] = g1 + a1;
    pv[2u] = g2 + a2;
    pv[3u] = g3 + a3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k][k2];
                }
            }
        }
    }
}
