// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let row = t + (idx >> 6u);
            let col = idx & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (row < cfg.m) {
                if (wid.x * 64u + col < cfg.d) {
                    va = phi[row * cfg.d + wid.x * 64u + col];
                }
                if (wid.y * 64u + col < cfg.r_rows && cfg.r_off + wid.y * 64u + col < cfg.d) {
                    vb = phi[row * cfg.d + cfg.r_off + wid.y * 64u + col];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            let o0 = k * 64u;
            let o1 = o0 + 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let b0 = wg_b[o0 + lid.y + 16u * i];
                let b1 = wg_b[o1 + lid.y + 16u * i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let a0 = wg_a[o0 + lid.x + 16u * j];
                    let a1 = wg_a[o1 + lid.x + 16u * j];
                    pb[i * 4u + j] = pb[i * 4u + j] + (b0 * a0 + b1 * a1);
                }
            }
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = wid.x * 64u + lid.x + 16u * k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 4u + k2];
                }
            }
        }
    }
}
