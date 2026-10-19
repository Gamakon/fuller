// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let c4 = (idx & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                let c = colbase + c4;
                if (c < d) { va.x = phi[base + c]; }
                if (c + 1u < d) { va.y = phi[base + c + 1u]; }
                if (c + 2u < d) { va.z = phi[base + c + 2u]; }
                if (c + 3u < d) { va.w = phi[base + c + 3u]; }
                let lr = rowbase + c4;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { vb.x = phi[base + gr]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < d) { vb.y = phi[base + gr + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < d) { vb.z = phi[base + gr + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < d) { vb.w = phi[base + gr + 3u]; }
            }
            wg_va[idx & 511u] = va;
            wg_vb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            let a = wg_va[(s * 16u + lid.x) & 511u];
            let b0 = wg_vb[(s * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(s * 16u + lid.y * 2u + 1u) & 511u];
            a0 = a0 + vec4<f32>(b0.x) * a;
            a1 = a1 + vec4<f32>(b0.y) * a;
            a2 = a2 + vec4<f32>(b0.z) * a;
            a3 = a3 + vec4<f32>(b0.w) * a;
            a4 = a4 + vec4<f32>(b1.x) * a;
            a5 = a5 + vec4<f32>(b1.y) * a;
            a6 = a6 + vec4<f32>(b1.z) * a;
            a7 = a7 + vec4<f32>(b1.w) * a;
        }
        workgroupBarrier();
    }
    var res: array<vec4<f32>, 8>;
    res[0u] = a0; res[1u] = a1; res[2u] = a2; res[3u] = a3;
    res[4u] = a4; res[5u] = a5; res[6u] = a6; res[7u] = a7;
    let c0 = colbase + lid.x * 4u;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let lr = rowbase + lid.y * 8u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let rv = res[j & 7u];
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + rv.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + rv.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + rv.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + rv.w; }
        }
    }
}
