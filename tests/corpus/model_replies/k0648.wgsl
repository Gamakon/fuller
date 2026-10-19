// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var r0: vec4<f32> = vec4<f32>(0.0);
    var r1: vec4<f32> = vec4<f32>(0.0);
    var r2: vec4<f32> = vec4<f32>(0.0);
    var r3: vec4<f32> = vec4<f32>(0.0);
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let it = t + (idx >> 4u);
            let off = (idx & 15u) * 4u;
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
            wg_va[idx] = a;
            wg_vb[idx] = b;
        }
        workgroupBarrier();
        var c0v = vec4<f32>(0.0);
        var c1v = vec4<f32>(0.0);
        var c2v = vec4<f32>(0.0);
        var c3v = vec4<f32>(0.0);
        var e0v = vec4<f32>(0.0);
        var e1v = vec4<f32>(0.0);
        var e2v = vec4<f32>(0.0);
        var e3v = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let i0 = k * 32u;
            let a0 = wg_va[i0 + lid.x];
            let b0 = wg_vb[i0 + lid.y];
            let a1 = wg_va[i0 + 16u + lid.x];
            let b1 = wg_vb[i0 + 16u + lid.y];
            c0v = c0v + vec4<f32>(b0.x) * a0;
            c1v = c1v + vec4<f32>(b0.y) * a0;
            c2v = c2v + vec4<f32>(b0.z) * a0;
            c3v = c3v + vec4<f32>(b0.w) * a0;
            e0v = e0v + vec4<f32>(b1.x) * a1;
            e1v = e1v + vec4<f32>(b1.y) * a1;
            e2v = e2v + vec4<f32>(b1.z) * a1;
            e3v = e3v + vec4<f32>(b1.w) * a1;
        }
        r0 = r0 + (c0v + e0v);
        r1 = r1 + (c1v + e1v);
        r2 = r2 + (c2v + e2v);
        r3 = r3 + (c3v + e3v);
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0u] = r0;
    pv[1u] = r1;
    pv[2u] = r2;
    pv[3u] = r3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + pv[k][k2];
                }
            }
        }
    }
}
