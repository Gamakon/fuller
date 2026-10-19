// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    var mid: array<vec4<f32>, 4>;
    var fin: array<vec4<f32>, 4>;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        acc[k] = vec4<f32>(0.0);
        mid[k] = vec4<f32>(0.0);
        fin[k] = vec4<f32>(0.0);
    }
    var cnt: u32 = 0u;
    let rowbase = wid.y * 64u;
    let colbase = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let row = idx / 16u;
            let q = idx % 16u;
            let item = t + row;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let rr = rowbase + q + 16u * j;
                    if (rr < cfg.r_rows && (cfg.r_off + rr) < cfg.d) {
                        v0[j] = phi[item * cfg.d + cfg.r_off + rr];
                    }
                    let cc = colbase + q + 16u * j;
                    if (cc < cfg.d) {
                        v1[j] = phi[item * cfg.d + cc];
                    }
                }
            }
            wg_va[idx] = v0;
            wg_vb[idx] = v1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            acc[k] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.y];
            let b = wg_vb[k * 16u + lid.x];
            acc[0] = acc[0] + vec4<f32>(a.x) * b;
            acc[1] = acc[1] + vec4<f32>(a.y) * b;
            acc[2] = acc[2] + vec4<f32>(a.z) * b;
            acc[3] = acc[3] + vec4<f32>(a.w) * b;
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            mid[k] = mid[k] + acc[k];
        }
        cnt = cnt + 1u;
        if (cnt == 16u) {
            for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                fin[k] = fin[k] + mid[k];
                mid[k] = vec4<f32>(0.0);
            }
            cnt = 0u;
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        fin[k] = fin[k] + mid[k];
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y + 16u * i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = colbase + lid.x + 16u * j;
            if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + fin[i][j];
            }
        }
    }
}
