// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var o0: vec4<f32> = vec4<f32>(0.0);
    var o1: vec4<f32> = vec4<f32>(0.0);
    var o2: vec4<f32> = vec4<f32>(0.0);
    var o3: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let e = lidx + 256u * q;
            let it = t + (e >> 4u);
            let vo = (e & 15u) * 4u;
            var c0: f32 = 0.0; var c1: f32 = 0.0; var c2: f32 = 0.0; var c3: f32 = 0.0;
            var r0: f32 = 0.0; var r1: f32 = 0.0; var r2: f32 = 0.0; var r3: f32 = 0.0;
            if (it < m) {
                let base = it * d;
                let cc = wid.x * 64u + vo;
                if (cc < d) { c0 = phi[base + cc]; }
                if (cc + 1u < d) { c1 = phi[base + cc + 1u]; }
                if (cc + 2u < d) { c2 = phi[base + cc + 2u]; }
                if (cc + 3u < d) { c3 = phi[base + cc + 3u]; }
                let lrr = wid.y * 64u + vo;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) { r0 = phi[base + gr]; }
                if (lrr + 1u < cfg.r_rows && gr + 1u < d) { r1 = phi[base + gr + 1u]; }
                if (lrr + 2u < cfg.r_rows && gr + 2u < d) { r2 = phi[base + gr + 2u]; }
                if (lrr + 3u < cfg.r_rows && gr + 3u < d) { r3 = phi[base + gr + 3u]; }
            }
            wg_va[e] = vec4<f32>(c0, c1, c2, c3);
            wg_vb[e] = vec4<f32>(r0, r1, r2, r3);
        }
        workgroupBarrier();
        var a0: vec4<f32> = vec4<f32>(0.0);
        var a1: vec4<f32> = vec4<f32>(0.0);
        var a2: vec4<f32> = vec4<f32>(0.0);
        var a3: vec4<f32> = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let bc = wg_va[k * 16u + lid.x];
            let ar = wg_vb[k * 16u + lid.y];
            a0 = a0 + ar.x * bc;
            a1 = a1 + ar.y * bc;
            a2 = a2 + ar.z * bc;
            a3 = a3 + ar.w * bc;
        }
        o0 = o0 + a0;
        o1 = o1 + a1;
        o2 = o2 + a2;
        o3 = o3 + a3;
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4> = array<vec4<f32>, 4>(o0, o1, o2, o3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let cb = wid.x * 64u + lid.x * 4u;
            let v = pv[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                if (cb + j < d) {
                    let idx = lr * d + cb + j;
                    h[idx] = h[idx] + v[j];
                }
            }
        }
    }
}
