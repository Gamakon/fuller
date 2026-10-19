// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var l1: array<vec4<f32>, 4>;
    var l2: array<vec4<f32>, 4>;
    var p: array<vec4<f32>, 4>;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        l1[k] = vec4<f32>(0.0);
        l2[k] = vec4<f32>(0.0);
    }
    var cnt: u32 = 0u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t + idx / 16u;
            let cg = idx % 16u;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let row = wid.y * 64u + cg * 4u + j;
                    if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
                        v0[j] = phi[base + cfg.r_off + row];
                    }
                    let col = wid.x * 64u + cg * 4u + j;
                    if (col < cfg.d) {
                        v1[j] = phi[base + col];
                    }
                }
            }
            wg_va[idx] = v0;
            wg_vb[idx] = v1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            p[k] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.y];
            let b = wg_vb[k * 16u + lid.x];
            p[0] = p[0] + vec4<f32>(a.x) * b;
            p[1] = p[1] + vec4<f32>(a.y) * b;
            p[2] = p[2] + vec4<f32>(a.z) * b;
            p[3] = p[3] + vec4<f32>(a.w) * b;
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            l1[k] = l1[k] + p[k];
        }
        cnt = cnt + 1u;
        if (cnt == 16u) {
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                l2[k] = l2[k] + l1[k];
                l1[k] = vec4<f32>(0.0);
            }
            cnt = 0u;
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        l2[k] = l2[k] + l1[k];
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = gid.y * 4u + k;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let col = gid.x * 4u + j;
            if (row < cfg.r_rows && col < cfg.d && cfg.r_off + row < cfg.d) {
                let o = row * cfg.d + col;
                h[o] = h[o] + l2[k][j];
            }
        }
    }
}
