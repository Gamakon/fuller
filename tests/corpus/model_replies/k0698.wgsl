// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var q: u32 = 0u; q < 8u; q = q + 1u) {
        acc[q] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let idx = lidx + 128u * j;
            let item = t + (idx >> 4u);
            let cv = (idx & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    let c = colBase + cv + e;
                    if (c < d) {
                        a[e] = phi[base + c];
                    }
                    let lr = rowBase + cv + e;
                    if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                        b[e] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx] = a;
            wg_vb[idx] = b;
        }
        workgroupBarrier();
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        var s4 = vec4<f32>(0.0); var s5 = vec4<f32>(0.0); var s6 = vec4<f32>(0.0); var s7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let o = k * 16u;
            let a0 = wg_va[o + lid.x];
            let a1 = wg_va[o + 8u + lid.x];
            let x = wg_vb[o + lid.y];
            s0 = s0 + vec4<f32>(x.x) * a0;
            s1 = s1 + vec4<f32>(x.y) * a0;
            s2 = s2 + vec4<f32>(x.z) * a0;
            s3 = s3 + vec4<f32>(x.w) * a0;
            s4 = s4 + vec4<f32>(x.x) * a1;
            s5 = s5 + vec4<f32>(x.y) * a1;
            s6 = s6 + vec4<f32>(x.z) * a1;
            s7 = s7 + vec4<f32>(x.w) * a1;
        }
        acc[0] = acc[0] + s0; acc[1] = acc[1] + s1; acc[2] = acc[2] + s2; acc[3] = acc[3] + s3;
        acc[4] = acc[4] + s4; acc[5] = acc[5] + s5; acc[6] = acc[6] + s6; acc[7] = acc[7] + s7;
        workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let lr = rowBase + lid.y * 4u + r;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var hv: u32 = 0u; hv < 2u; hv = hv + 1u) {
                let cb = colBase + hv * 32u + lid.x * 4u;
                let v = acc[hv * 4u + r];
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    let c = cb + e;
                    if (c < d) {
                        let ix = lr * d + c;
                        h[ix] = h[ix] + v[e];
                    }
                }
            }
        }
    }
}
