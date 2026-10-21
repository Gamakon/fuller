// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var accA: array<vec4<f32>, 8>;
    var accB: array<vec4<f32>, 8>;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        accA[j] = vec4<f32>(0.0);
        accB[j] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 128u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = lidx + 256u * k;
            let item = t + (e >> 5u);
            let col = colBase + (e & 31u) * 4u;
            var v = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d + col;
                if (col < d) { v.x = phi[base]; }
                if (col + 1u < d) { v.y = phi[base + 1u]; }
                if (col + 2u < d) { v.z = phi[base + 2u]; }
                if (col + 3u < d) { v.w = phi[base + 3u]; }
            }
            wg_va[e] = v;
        }
        {
            let e = lidx;
            let item = t + (e >> 4u);
            let rl = rowBase + (e & 15u) * 4u;
            var v = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d + cfg.r_off + rl;
                if (rl < cfg.r_rows && cfg.r_off + rl < d) { v.x = phi[base]; }
                if (rl + 1u < cfg.r_rows && cfg.r_off + rl + 1u < d) { v.y = phi[base + 1u]; }
                if (rl + 2u < cfg.r_rows && cfg.r_off + rl + 2u < d) { v.z = phi[base + 2u]; }
                if (rl + 3u < cfg.r_rows && cfg.r_off + rl + 3u < d) { v.w = phi[base + 3u]; }
            }
            wg_vb[e] = v;
        }
        workgroupBarrier();
        var t0 = vec4<f32>(0.0);
        var t1 = vec4<f32>(0.0);
        var t2 = vec4<f32>(0.0);
        var t3 = vec4<f32>(0.0);
        var t4 = vec4<f32>(0.0);
        var t5 = vec4<f32>(0.0);
        var t6 = vec4<f32>(0.0);
        var t7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_va[k * 32u + lid.x];
            let b0 = wg_vb[k * 16u + lid.y * 2u];
            let b1 = wg_vb[k * 16u + lid.y * 2u + 1u];
            t0 = t0 + vec4<f32>(b0.x) * a;
            t1 = t1 + vec4<f32>(b0.y) * a;
            t2 = t2 + vec4<f32>(b0.z) * a;
            t3 = t3 + vec4<f32>(b0.w) * a;
            t4 = t4 + vec4<f32>(b1.x) * a;
            t5 = t5 + vec4<f32>(b1.y) * a;
            t6 = t6 + vec4<f32>(b1.z) * a;
            t7 = t7 + vec4<f32>(b1.w) * a;
        }
        if (((t >> 4u) & 1u) == 0u) {
            accA[0] = accA[0] + t0; accA[1] = accA[1] + t1; accA[2] = accA[2] + t2; accA[3] = accA[3] + t3;
            accA[4] = accA[4] + t4; accA[5] = accA[5] + t5; accA[6] = accA[6] + t6; accA[7] = accA[7] + t7;
        } else {
            accB[0] = accB[0] + t0; accB[1] = accB[1] + t1; accB[2] = accB[2] + t2; accB[3] = accB[3] + t3;
            accB[4] = accB[4] + t4; accB[5] = accB[5] + t5; accB[6] = accB[6] + t6; accB[7] = accB[7] + t7;
        }
        workgroupBarrier();
    }
    let c0 = (wid.x * 32u + lid.x) * 4u;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let lr = (wid.y * 8u + lid.y) * 8u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let s = accA[j] + accB[j];
            for (var cc: u32 = 0u; cc < 4u; cc = cc + 1u) {
                if (c0 + cc < d) {
                    let idx = lr * d + c0 + cc;
                    h[idx] = h[idx] + s[cc];
                }
            }
        }
    }
}
