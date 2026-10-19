// grid 32 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var v10: vec4<f32> = vec4<f32>(0.0);
    var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0);
    var v13: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 5u);
            let col = wid.x * 128u + (idx & 31u) * 4u;
            var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
            if (item < cfg.m) {
                let base = item * cfg.d;
                if (col < cfg.d) { a0 = phi[base + col]; }
                if (col + 1u < cfg.d) { a1 = phi[base + col + 1u]; }
                if (col + 2u < cfg.d) { a2 = phi[base + col + 2u]; }
                if (col + 3u < cfg.d) { a3 = phi[base + col + 3u]; }
            }
            wg_va[idx] = vec4<f32>(a0, a1, a2, a3);
        }
        if (lidx < 128u) {
            let item = t + (lidx >> 3u);
            let row = wid.y * 32u + (lidx & 7u) * 4u;
            var b0: f32 = 0.0; var b1: f32 = 0.0; var b2: f32 = 0.0; var b3: f32 = 0.0;
            if (item < cfg.m) {
                let base = item * cfg.d + cfg.r_off + row;
                if (row < cfg.r_rows && cfg.r_off + row < cfg.d) { b0 = phi[base]; }
                if (row + 1u < cfg.r_rows && cfg.r_off + row + 1u < cfg.d) { b1 = phi[base + 1u]; }
                if (row + 2u < cfg.r_rows && cfg.r_off + row + 2u < cfg.d) { b2 = phi[base + 2u]; }
                if (row + 3u < cfg.r_rows && cfg.r_off + row + 3u < cfg.d) { b3 = phi[base + 3u]; }
            }
            wg_vb[lidx] = vec4<f32>(b0, b1, b2, b3);
        }
        workgroupBarrier();
        v0 = vec4<f32>(0.0);
        v1 = vec4<f32>(0.0);
        v2 = vec4<f32>(0.0);
        v3 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            v4 = wg_va[k * 32u + lid.x];
            v5 = wg_vb[k * 8u + lid.y];
            v0 = v0 + vec4<f32>(v5.x) * v4;
            v1 = v1 + vec4<f32>(v5.y) * v4;
            v2 = v2 + vec4<f32>(v5.z) * v4;
            v3 = v3 + vec4<f32>(v5.w) * v4;
        }
        v6 = v6 + v0;
        v7 = v7 + v1;
        v8 = v8 + v2;
        v9 = v9 + v3;
        if (((t >> 4u) & 31u) == 31u) {
            v10 = v10 + v6;
            v11 = v11 + v7;
            v12 = v12 + v8;
            v13 = v13 + v9;
            v6 = vec4<f32>(0.0);
            v7 = vec4<f32>(0.0);
            v8 = vec4<f32>(0.0);
            v9 = vec4<f32>(0.0);
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0u] = v10 + v6;
    pv[1u] = v11 + v7;
    pv[2u] = v12 + v8;
    pv[3u] = v13 + v9;
    for (k = 0u; k < 4u; k = k + 1u) {
        let row = wid.y * 32u + lid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 128u + lid.x * 4u + j;
                if (col < cfg.d) {
                    let o = row * cfg.d + col;
                    h[o] = h[o] + pv[k][j];
                }
            }
        }
    }
}
