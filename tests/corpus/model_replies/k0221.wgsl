// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var e0 = vec4<f32>(0.0); var e1 = vec4<f32>(0.0); var e2 = vec4<f32>(0.0); var e3 = vec4<f32>(0.0);
    var o0 = vec4<f32>(0.0); var o1 = vec4<f32>(0.0); var o2 = vec4<f32>(0.0); var o3 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        // fused tile load of both operands
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let li = lidx + 256u * k;
            let item = t + (li >> 4u);
            let c4 = (li & 15u) * 4u;
            let ca = wid.x * 64u + c4;
            let rl = wid.y * 64u + c4;
            let rg = cfg.r_off + rl;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                if (ca < d) { a.x = phi[base + ca]; }
                if (ca + 1u < d) { a.y = phi[base + ca + 1u]; }
                if (ca + 2u < d) { a.z = phi[base + ca + 2u]; }
                if (ca + 3u < d) { a.w = phi[base + ca + 3u]; }
                if (rl < cfg.r_rows && rg < d) { b.x = phi[base + rg]; }
                if (rl + 1u < cfg.r_rows && rg + 1u < d) { b.y = phi[base + rg + 1u]; }
                if (rl + 2u < cfg.r_rows && rg + 2u < d) { b.z = phi[base + rg + 2u]; }
                if (rl + 3u < cfg.r_rows && rg + 3u < d) { b.w = phi[base + rg + 3u]; }
            }
            wg_va[li & 511u] = a;
            wg_vb[li & 511u] = b;
        }
        workgroupBarrier();
        // fused compute: two items per iteration, accumulating directly across tiles
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let i0 = (2u * k) * 16u;
            let i1 = i0 + 16u;
            let xa = wg_va[(i0 + lid.x) & 511u];
            let yb = wg_vb[(i0 + lid.y) & 511u];
            let xc = wg_va[(i1 + lid.x) & 511u];
            let yd = wg_vb[(i1 + lid.y) & 511u];
            e0 = e0 + vec4<f32>(yb.x) * xa;
            e1 = e1 + vec4<f32>(yb.y) * xa;
            e2 = e2 + vec4<f32>(yb.z) * xa;
            e3 = e3 + vec4<f32>(yb.w) * xa;
            o0 = o0 + vec4<f32>(yd.x) * xc;
            o1 = o1 + vec4<f32>(yd.y) * xc;
            o2 = o2 + vec4<f32>(yd.z) * xc;
            o3 = o3 + vec4<f32>(yd.w) * xc;
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0u] = e0 + o0;
    pv[1u] = e1 + o1;
    pv[2u] = e2 + o2;
    pv[3u] = e3 + o3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = gid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < d) {
            let v = pv[k & 3u];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = gid.x * 4u + k2;
                if (col < d) {
                    let idx = row * d + col;
                    h[idx] = h[idx] + v[k2 & 3u];
                }
            }
        }
    }
}
