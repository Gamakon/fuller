// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var p0 = vec4<f32>(0.0);
    var p1 = vec4<f32>(0.0);
    var p2 = vec4<f32>(0.0);
    var p3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let idx = lidx + 256u * k2;
            let item = t + (idx >> 4u);
            let cq = (idx & 15u) * 4u;
            var fa = vec4<f32>(0.0);
            var fb = vec4<f32>(0.0);
            if (item < cfg.m) {
                let rowbase = item * cfg.d;
                let c0 = wid.x * 64u + cq;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (c0 + j < cfg.d) {
                        fa[j] = phi[rowbase + c0 + j];
                    }
                    let lr = wid.y * 64u + cq + j;
                    if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                        fb[j] = phi[rowbase + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx & 511u] = fa;
            wg_vb[idx & 511u] = fb;
        }
        workgroupBarrier();
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let va = wg_va[(k * 16u + lid.x) & 511u];
            let vb = wg_vb[(k * 16u + lid.y) & 511u];
            a0 = a0 + vec4<f32>(vb.x) * va;
            a1 = a1 + vec4<f32>(vb.y) * va;
            a2 = a2 + vec4<f32>(vb.z) * va;
            a3 = a3 + vec4<f32>(vb.w) * va;
        }
        p0 = p0 + a0;
        p1 = p1 + a1;
        p2 = p2 + a2;
        p3 = p3 + a3;
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = gid.y * 4u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            var v = p0;
            if (j == 1u) { v = p1; }
            if (j == 2u) { v = p2; }
            if (j == 3u) { v = p3; }
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let c = gid.x * 4u + q;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + v[q];
                }
            }
        }
    }
}
