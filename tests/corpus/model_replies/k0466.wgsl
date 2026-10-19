// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e / 64u;
            let cc = e % 64u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t + kk < cfg.m) {
                let lr = row0 + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                    va = phi[(t + kk) * cfg.d + cfg.r_off + lr];
                }
                if (col0 + cc < cfg.d) {
                    vb = phi[(t + kk) * cfg.d + col0 + cc];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                ra[i] = wg_a[k * 64u + lid.y + 16u * i];
                rb[i] = wg_b[k * 64u + lid.x + 16u * i];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + ra[i] * rb[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
