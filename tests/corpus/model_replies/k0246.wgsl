// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        acc[i & 7u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 128u * k;
            let item = t + (e >> 4u);
            let q = (e & 15u) * 4u;
            let cb = wid.x * 64u + q;
            let rb = wid.y * 64u + q;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (cb + j < d) {
                        va[j] = phi[base + cb + j];
                    }
                    if (rb + j < cfg.r_rows && cfg.r_off + rb + j < d) {
                        vb[j] = phi[base + cfg.r_off + rb + j];
                    }
                }
            }
            wg_va[e & 511u] = va;
            wg_vb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            let a = wg_va[(s * 16u + lid.x) & 511u];
            let b0 = wg_vb[(s * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(s * 16u + lid.y * 2u + 1u) & 511u];
            acc[0u] = acc[0u] + b0.x * a;
            acc[1u] = acc[1u] + b0.y * a;
            acc[2u] = acc[2u] + b0.z * a;
            acc[3u] = acc[3u] + b0.w * a;
            acc[4u] = acc[4u] + b1.x * a;
            acc[5u] = acc[5u] + b1.y * a;
            acc[6u] = acc[6u] + b1.z * a;
            acc[7u] = acc[7u] + b1.w * a;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 8u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[i & 7u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[j & 3u];
                }
            }
        }
    }
}
