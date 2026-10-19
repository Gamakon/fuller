// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let it = t + (idx >> 4u);
            let q = (idx & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (it < m) {
                let base = it * d;
                let c0 = colBase + q;
                if (c0 < d) { va.x = phi[base + c0]; }
                if (c0 + 1u < d) { va.y = phi[base + c0 + 1u]; }
                if (c0 + 2u < d) { va.z = phi[base + c0 + 2u]; }
                if (c0 + 3u < d) { va.w = phi[base + c0 + 3u]; }
                let l0 = rowBase + q;
                let g0 = r_off + l0;
                if (l0 < r_rows && g0 < d) { vb.x = phi[base + g0]; }
                if (l0 + 1u < r_rows && g0 + 1u < d) { vb.y = phi[base + g0 + 1u]; }
                if (l0 + 2u < r_rows && g0 + 2u < d) { vb.z = phi[base + g0 + 2u]; }
                if (l0 + 3u < r_rows && g0 + 3u < d) { vb.w = phi[base + g0 + 3u]; }
            }
            wg_va[idx & 511u] = va;
            wg_vb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b = wg_vb[(k * 16u + lid.y) & 511u];
            a0 = a0 + vec4<f32>(b.x) * a;
            a1 = a1 + vec4<f32>(b.y) * a;
            a2 = a2 + vec4<f32>(b.z) * a;
            a3 = a3 + vec4<f32>(b.w) * a;
        }
        workgroupBarrier();
    }
    var acc: array<vec4<f32>, 4>;
    acc[0u] = a0;
    acc[1u] = a1;
    acc[2u] = a2;
    acc[3u] = a3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = rowBase + lid.y * 4u + k;
        if (lr < r_rows && r_off + lr < d) {
            let row = acc[k & 3u];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = colBase + lid.x * 4u + k2;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + row[k2 & 3u];
                }
            }
        }
    }
}
