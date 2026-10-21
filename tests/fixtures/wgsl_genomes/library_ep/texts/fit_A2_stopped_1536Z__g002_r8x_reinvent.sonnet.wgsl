// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    var e0 = vec4<f32>(0.0); var e1 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0); var e3 = vec4<f32>(0.0);
    var e4 = vec4<f32>(0.0); var e5 = vec4<f32>(0.0); var e6 = vec4<f32>(0.0); var e7 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i = lidx + 256u * k;
            let tt = t + (i >> 5u);
            let col = wid.x * 128u + (i & 31u) * 4u;
            var x = vec4<f32>(0.0);
            if (tt < m) {
                let base = tt * d;
                if (col < d) { x.x = phi[base + col]; }
                if (col + 1u < d) { x.y = phi[base + col + 1u]; }
                if (col + 2u < d) { x.z = phi[base + col + 2u]; }
                if (col + 3u < d) { x.w = phi[base + col + 3u]; }
            }
            wg_va[i] = x;
        }
        {
            let tt = t + (lidx >> 4u);
            let row = wid.y * 64u + (lidx & 15u) * 4u;
            var y = vec4<f32>(0.0);
            if (tt < m) {
                let base = tt * d + cfg.r_off + row;
                if (row < cfg.r_rows && cfg.r_off + row < d) { y.x = phi[base]; }
                if (row + 1u < cfg.r_rows && cfg.r_off + row + 1u < d) { y.y = phi[base + 1u]; }
                if (row + 2u < cfg.r_rows && cfg.r_off + row + 2u < d) { y.z = phi[base + 2u]; }
                if (row + 3u < cfg.r_rows && cfg.r_off + row + 3u < d) { y.w = phi[base + 3u]; }
            }
            wg_vb[lidx] = y;
        }
        workgroupBarrier();
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        var s4 = vec4<f32>(0.0); var s5 = vec4<f32>(0.0); var s6 = vec4<f32>(0.0); var s7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_va[k * 32u + lid.x * 2u];
            let a1 = wg_va[k * 32u + lid.x * 2u + 1u];
            let b = wg_vb[k * 16u + lid.y];
            s0 = s0 + vec4<f32>(b.x) * a0; s1 = s1 + vec4<f32>(b.x) * a1;
            s2 = s2 + vec4<f32>(b.y) * a0; s3 = s3 + vec4<f32>(b.y) * a1;
            s4 = s4 + vec4<f32>(b.z) * a0; s5 = s5 + vec4<f32>(b.z) * a1;
            s6 = s6 + vec4<f32>(b.w) * a0; s7 = s7 + vec4<f32>(b.w) * a1;
        }
        e0 = e0 + s0; e1 = e1 + s1; e2 = e2 + s2; e3 = e3 + s3;
        e4 = e4 + s4; e5 = e5 + s5; e6 = e6 + s6; e7 = e7 + s7;
        workgroupBarrier();
    }
    pv[0] = e0; pv[1] = e1; pv[2] = e2; pv[3] = e3;
    pv[4] = e4; pv[5] = e5; pv[6] = e6; pv[7] = e7;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = wid.y * 64u + lid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let col = wid.x * 128u + lid.x * 8u + j;
                if (col < d) {
                    let idx = row * d + col;
                    h[idx] = h[idx] + pv[k * 2u + (j >> 2u)][j & 3u];
                }
            }
        }
    }
}
