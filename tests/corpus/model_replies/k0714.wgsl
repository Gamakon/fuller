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
    var pv: array<vec4<f32>, 16>;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let row = t + (idx >> 4u);
            let cc = (idx & 15u) * 4u;
            let ca = wid.x * 64u + cc;
            let cb = wid.y * 64u + cc;
            var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
            var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
            if (row < cfg.m) {
                let base = row * cfg.d;
                if (ca < cfg.d) { f0 = phi[base + ca]; }
                if (ca + 1u < cfg.d) { f1 = phi[base + ca + 1u]; }
                if (ca + 2u < cfg.d) { f2 = phi[base + ca + 2u]; }
                if (ca + 3u < cfg.d) { f3 = phi[base + ca + 3u]; }
                let rb = base + cfg.r_off + cb;
                if (cb < cfg.r_rows && cfg.r_off + cb < cfg.d) { f4 = phi[rb]; }
                if (cb + 1u < cfg.r_rows && cfg.r_off + cb + 1u < cfg.d) { f5 = phi[rb + 1u]; }
                if (cb + 2u < cfg.r_rows && cfg.r_off + cb + 2u < cfg.d) { f6 = phi[rb + 2u]; }
                if (cb + 3u < cfg.r_rows && cfg.r_off + cb + 3u < cfg.d) { f7 = phi[rb + 3u]; }
            }
            wg_va[idx & 511u] = vec4<f32>(f0, f1, f2, f3);
            wg_vb[idx & 511u] = vec4<f32>(f4, f5, f6, f7);
        }
        workgroupBarrier();
        var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
        var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ia = (2u * k) * 16u;
            let xa = wg_va[(ia + lid.x) & 511u];
            let ya = wg_vb[(ia + lid.y) & 511u];
            a0 = a0 + vec4<f32>(ya.x) * xa;
            a1 = a1 + vec4<f32>(ya.y) * xa;
            a2 = a2 + vec4<f32>(ya.z) * xa;
            a3 = a3 + vec4<f32>(ya.w) * xa;
            let ib = ia + 16u;
            let xb = wg_va[(ib + lid.x) & 511u];
            let yb = wg_vb[(ib + lid.y) & 511u];
            b0 = b0 + vec4<f32>(yb.x) * xb;
            b1 = b1 + vec4<f32>(yb.y) * xb;
            b2 = b2 + vec4<f32>(yb.z) * xb;
            b3 = b3 + vec4<f32>(yb.w) * xb;
        }
        if (((t >> 5u) & 1u) == 0u) {
            e0 = e0 + (a0 + b0);
            e1 = e1 + (a1 + b1);
            e2 = e2 + (a2 + b2);
            e3 = e3 + (a3 + b3);
        } else {
            o0 = o0 + (a0 + b0);
            o1 = o1 + (a1 + b1);
            o2 = o2 + (a2 + b2);
            o3 = o3 + (a3 + b3);
        }
        workgroupBarrier();
    }
    pv[0u] = e0 + o0;
    pv[1u] = e1 + o1;
    pv[2u] = e2 + o2;
    pv[3u] = e3 + o3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pv[k][k2];
                }
            }
        }
    }
}
