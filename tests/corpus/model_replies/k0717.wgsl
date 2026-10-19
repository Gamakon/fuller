// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var o0 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var o1 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let d = cfg.d;
    let m = cfg.m;
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 4u);
            let c4 = (idx & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < m) {
                let rowp = item * d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (cbase + c4 + j < d) {
                        va[j] = phi[rowp + cbase + c4 + j];
                    }
                    if (rbase + c4 + j < cfg.r_rows && cfg.r_off + rbase + c4 + j < d) {
                        vb[j] = phi[rowp + cfg.r_off + rbase + c4 + j];
                    }
                }
            }
            wg_va[idx] = va;
            wg_vb[idx] = vb;
        }
        workgroupBarrier();
        var s0 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        var s1 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_va[(2u * k) * 16u + lid.x];
            let b0 = wg_vb[(2u * k) * 16u + lid.y];
            let a1 = wg_va[(2u * k + 1u) * 16u + lid.x];
            let b1 = wg_vb[(2u * k + 1u) * 16u + lid.y];
            s0[0] = s0[0] + vec4<f32>(b0.x) * a0;
            s0[1] = s0[1] + vec4<f32>(b0.y) * a0;
            s0[2] = s0[2] + vec4<f32>(b0.z) * a0;
            s0[3] = s0[3] + vec4<f32>(b0.w) * a0;
            s1[0] = s1[0] + vec4<f32>(b1.x) * a1;
            s1[1] = s1[1] + vec4<f32>(b1.y) * a1;
            s1[2] = s1[2] + vec4<f32>(b1.z) * a1;
            s1[3] = s1[3] + vec4<f32>(b1.w) * a1;
        }
        if (((t >> 5u) & 1u) == 0u) {
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                o0[k] = o0[k] + (s0[k] + s1[k]);
            }
        } else {
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                o1[k] = o1[k] + (s0[k] + s1[k]);
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let sum = o0[k] + o1[k];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < d) {
                    h[lr * d + c] = h[lr * d + c] + sum[k2];
                }
            }
        }
    }
}
