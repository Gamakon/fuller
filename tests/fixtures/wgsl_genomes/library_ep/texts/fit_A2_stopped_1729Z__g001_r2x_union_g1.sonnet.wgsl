// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var o0 = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var o1 = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var s0 = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var s1 = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var tmp = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let d = cfg.d;
    let m = cfg.m;
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 4u);
            let c4 = (idx & 15u) * 4u;
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                tmp[j] = 0.0;
            }
            if (item < m) {
                let rowp = item * d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (cbase + c4 + j < d) {
                        tmp[j] = phi[rowp + cbase + c4 + j];
                    }
                    if (rbase + c4 + j < cfg.r_rows) {
                        if (cfg.r_off + rbase + c4 + j < d) {
                            tmp[j + 4u] = phi[rowp + cfg.r_off + rbase + c4 + j];
                        }
                    }
                }
            }
            wg_va[idx] = vec4<f32>(tmp[0], tmp[1], tmp[2], tmp[3]);
            wg_vb[idx] = vec4<f32>(tmp[4], tmp[5], tmp[6], tmp[7]);
        }
        workgroupBarrier();
        for (var q: u32 = 0u; q < 16u; q = q + 1u) {
            s0[q] = 0.0;
            s1[q] = 0.0;
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_va[(2u * k) * 16u + lid.x];
            let b0 = wg_vb[(2u * k) * 16u + lid.y];
            let a1 = wg_va[(2u * k + 1u) * 16u + lid.x];
            let b1 = wg_vb[(2u * k + 1u) * 16u + lid.y];
            for (var r: u32 = 0u; r < 4u; r = r + 1u) {
                for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                    s0[r * 4u + c] = s0[r * 4u + c] + b0[r] * a0[c];
                    s1[r * 4u + c] = s1[r * 4u + c] + b1[r] * a1[c];
                }
            }
        }
        if (((t >> 5u) & 1u) == 0u) {
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                o0[q] = o0[q] + (s0[q] + s1[q]);
            }
        } else {
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                o1[q] = o1[q] + (s0[q] + s1[q]);
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows) {
            if (cfg.r_off + lr < d) {
                for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    let c = gid.x * 4u + k2;
                    if (c < d) {
                        h[lr * d + c] = h[lr * d + c] + (o0[k * 4u + k2] + o1[k * 4u + k2]);
                    }
                }
            }
        }
    }
}
