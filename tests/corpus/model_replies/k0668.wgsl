// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    var p0 = vec4<f32>(0.0); var p1 = vec4<f32>(0.0); var p2 = vec4<f32>(0.0); var p3 = vec4<f32>(0.0);
    var p4 = vec4<f32>(0.0); var p5 = vec4<f32>(0.0); var p6 = vec4<f32>(0.0); var p7 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let tb = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let li = lidx + 128u * k2;
                let item = tb + (li >> 4u);
                let cc = (li & 15u) * 4u;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                if (item < cfg.m) {
                    let rowbase = item * cfg.d;
                    let ca = wid.x * 64u + cc;
                    if (ca < cfg.d) { a.x = phi[rowbase + ca]; }
                    if (ca + 1u < cfg.d) { a.y = phi[rowbase + ca + 1u]; }
                    if (ca + 2u < cfg.d) { a.z = phi[rowbase + ca + 2u]; }
                    if (ca + 3u < cfg.d) { a.w = phi[rowbase + ca + 3u]; }
                    let lr0 = wid.y * 64u + cc;
                    let gb = rowbase + cfg.r_off + lr0;
                    if (lr0 < cfg.r_rows && cfg.r_off + lr0 < cfg.d) { b.x = phi[gb]; }
                    if (lr0 + 1u < cfg.r_rows && cfg.r_off + lr0 + 1u < cfg.d) { b.y = phi[gb + 1u]; }
                    if (lr0 + 2u < cfg.r_rows && cfg.r_off + lr0 + 2u < cfg.d) { b.z = phi[gb + 2u]; }
                    if (lr0 + 3u < cfg.r_rows && cfg.r_off + lr0 + 3u < cfg.d) { b.w = phi[gb + 3u]; }
                }
                wg_va[li] = a;
                wg_vb[li] = b;
            }
            workgroupBarrier();
            var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
            var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
            for (var j: u32 = 0u; j < 16u; j = j + 1u) {
                let i14 = j * 32u;
                let i15 = i14 + lid.y * 2u;
                let a0 = wg_va[i14 + lid.x];
                let a1 = wg_va[i14 + 16u + lid.x];
                let b0 = wg_vb[i15];
                let b1 = wg_vb[i15 + 1u];
                let b2 = wg_vb[i15 + 16u];
                let b3 = wg_vb[i15 + 17u];
                v0 = v0 + (vec4<f32>(b0.x) * a0 + vec4<f32>(b2.x) * a1);
                v1 = v1 + (vec4<f32>(b0.y) * a0 + vec4<f32>(b2.y) * a1);
                v2 = v2 + (vec4<f32>(b0.z) * a0 + vec4<f32>(b2.z) * a1);
                v3 = v3 + (vec4<f32>(b0.w) * a0 + vec4<f32>(b2.w) * a1);
                v4 = v4 + (vec4<f32>(b1.x) * a0 + vec4<f32>(b3.x) * a1);
                v5 = v5 + (vec4<f32>(b1.y) * a0 + vec4<f32>(b3.y) * a1);
                v6 = v6 + (vec4<f32>(b1.z) * a0 + vec4<f32>(b3.z) * a1);
                v7 = v7 + (vec4<f32>(b1.w) * a0 + vec4<f32>(b3.w) * a1);
            }
            p0 = p0 + v0; p1 = p1 + v1; p2 = p2 + v2; p3 = p3 + v3;
            p4 = p4 + v4; p5 = p5 + v5; p6 = p6 + v6; p7 = p7 + v7;
            workgroupBarrier();
        }
        pv[0] = pv[0] + p0; pv[1] = pv[1] + p1; pv[2] = pv[2] + p2; pv[3] = pv[3] + p3;
        pv[4] = pv[4] + p4; pv[5] = pv[5] + p5; pv[6] = pv[6] + p6; pv[7] = pv[7] + p7;
        p0 = vec4<f32>(0.0); p1 = vec4<f32>(0.0); p2 = vec4<f32>(0.0); p3 = vec4<f32>(0.0);
        p4 = vec4<f32>(0.0); p5 = vec4<f32>(0.0); p6 = vec4<f32>(0.0); p7 = vec4<f32>(0.0);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
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
