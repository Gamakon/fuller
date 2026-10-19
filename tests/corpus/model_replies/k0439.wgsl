// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let ntiles = (cfg.m + 31u) / 32u;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t * 32u + idx / 16u;
            let c4 = (idx % 16u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let lr = wid.y * 64u + c4 + j;
                    let gc = wid.x * 64u + c4 + j;
                    if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                        va[j] = phi[item * cfg.d + cfg.r_off + lr];
                    }
                    if (gc < cfg.d) {
                        vb[j] = phi[item * cfg.d + gc];
                    }
                }
            }
            wg_va[idx] = va;
            wg_vb[idx] = vb;
        }
        workgroupBarrier();
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var g = 0u; g < 4u; g = g + 1u) {
            var q0 = vec4<f32>(0.0);
            var q1 = vec4<f32>(0.0);
            var q2 = vec4<f32>(0.0);
            var q3 = vec4<f32>(0.0);
            for (var kk = 0u; kk < 8u; kk = kk + 1u) {
                let a = wg_va[(g * 8u + kk) * 16u + lid.y];
                let b = wg_vb[(g * 8u + kk) * 16u + lid.x];
                q0 = q0 + vec4<f32>(a.x) * b;
                q1 = q1 + vec4<f32>(a.y) * b;
                q2 = q2 + vec4<f32>(a.z) * b;
                q3 = q3 + vec4<f32>(a.w) * b;
            }
            s0 = s0 + q0;
            s1 = s1 + q1;
            s2 = s2 + q2;
            s3 = s3 + q3;
        }
        a0 = a0 + s0;
        a1 = a1 + s1;
        a2 = a2 + s2;
        a3 = a3 + s3;
        workgroupBarrier();
    }
    for (var k = 0u; k < 4u; k = k + 1u) {
        var v = a0;
        if (k == 1u) { v = a1; }
        if (k == 2u) { v = a2; }
        if (k == 3u) { v = a3; }
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + v[k2];
                }
            }
        }
    }
}
