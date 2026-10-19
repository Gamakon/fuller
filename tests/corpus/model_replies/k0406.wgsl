// grid 32 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 128>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    var cnt: u32 = 0u;
    let nt = (cfg.m + 15u) / 16u;
    for (var t: u32 = 0u; t < nt; t = t + 1u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t * 16u + idx / 32u;
            let col0 = wid.x * 128u + (idx % 32u) * 4u;
            var v = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (col0 + j < cfg.d) {
                        v[j] = phi[base + col0 + j];
                    }
                }
            }
            wg_vb[idx] = v;
        }
        if (lidx < 128u) {
            let item = t * 16u + lidx / 8u;
            let row0 = wid.y * 32u + (lidx % 8u) * 4u;
            var v = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d + cfg.r_off;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let rr = row0 + j;
                    if (rr < cfg.r_rows && cfg.r_off + rr < cfg.d) {
                        v[j] = phi[base + rr];
                    }
                }
            }
            wg_va[lidx] = v;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            pv[k] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_va[k * 8u + lid.y];
            let b = wg_vb[k * 32u + lid.x];
            pv[0] = pv[0] + vec4<f32>(a.x) * b;
            pv[1] = pv[1] + vec4<f32>(a.y) * b;
            pv[2] = pv[2] + vec4<f32>(a.z) * b;
            pv[3] = pv[3] + vec4<f32>(a.w) * b;
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            pv[4u + k] = pv[4u + k] + pv[k];
        }
        cnt = cnt + 1u;
        if (cnt == 16u) {
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                pv[8u + k] = pv[8u + k] + pv[4u + k];
                pv[4u + k] = vec4<f32>(0.0);
            }
            cnt = 0u;
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        pv[8u + k] = pv[8u + k] + pv[4u + k];
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let row = wid.y * 32u + lid.y * 4u + i;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 128u + lid.x * 4u + j;
                if (col < cfg.d) {
                    let o = row * cfg.d + col;
                    h[o] = h[o] + pv[8u + i][j];
                }
            }
        }
    }
}
