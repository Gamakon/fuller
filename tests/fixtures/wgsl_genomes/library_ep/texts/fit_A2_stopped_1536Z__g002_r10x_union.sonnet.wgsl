// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var e0 = vec4<f32>(0.0); var e1 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0); var e3 = vec4<f32>(0.0);
    var o0 = vec4<f32>(0.0); var o1 = vec4<f32>(0.0); var o2 = vec4<f32>(0.0); var o3 = vec4<f32>(0.0);
    var t: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i2 = lidx + 256u * k;
            let it = t + (i2 >> 4u);
            let c0 = wid.x * 64u + (i2 & 15u) * 4u;
            let r0 = wid.y * 64u + (i2 & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (it < cfg.m) {
                let base = it * cfg.d;
                if (c0 < cfg.d) { a.x = phi[base + c0]; }
                if (c0 + 1u < cfg.d) { a.y = phi[base + c0 + 1u]; }
                if (c0 + 2u < cfg.d) { a.z = phi[base + c0 + 2u]; }
                if (c0 + 3u < cfg.d) { a.w = phi[base + c0 + 3u]; }
                let g = base + cfg.r_off + r0;
                if (r0 < cfg.r_rows && cfg.r_off + r0 < cfg.d) { b.x = phi[g]; }
                if (r0 + 1u < cfg.r_rows && cfg.r_off + r0 + 1u < cfg.d) { b.y = phi[g + 1u]; }
                if (r0 + 2u < cfg.r_rows && cfg.r_off + r0 + 2u < cfg.d) { b.z = phi[g + 2u]; }
                if (r0 + 3u < cfg.r_rows && cfg.r_off + r0 + 3u < cfg.d) { b.w = phi[g + 3u]; }
            }
            wg_va[i2 & 511u] = a;
            wg_vb[i2 & 511u] = b;
        }
        workgroupBarrier();
        var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let x = wg_va[(k * 16u + lid.x) & 511u];
            let y = wg_vb[(k * 16u + lid.y) & 511u];
            v0 = v0 + vec4<f32>(y.x) * x;
            v1 = v1 + vec4<f32>(y.y) * x;
            v2 = v2 + vec4<f32>(y.z) * x;
            v3 = v3 + vec4<f32>(y.w) * x;
        }
        if (((t >> 5u) & 1u) == 0u) {
            e0 = e0 + v0; e1 = e1 + v1; e2 = e2 + v2; e3 = e3 + v3;
        } else {
            o0 = o0 + v0; o1 = o1 + v1; o2 = o2 + v2; o3 = o3 + v3;
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0] = e0 + o0;
    pv[1] = e1 + o1;
    pv[2] = e2 + o2;
    pv[3] = e3 + o3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = gid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = gid.x * 4u + k2;
                if (col < cfg.d) {
                    h[row * cfg.d + col] = h[row * cfg.d + col] + pv[k][k2];
                }
            }
        }
    }
}
