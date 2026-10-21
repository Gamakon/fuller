// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let e = lidx + 128u * k;
            let rr = e >> 6u;
            let cc = e & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t + rr < cfg.m) {
                if (col0 + cc < cfg.d) {
                    va = phi[(t + rr) * cfg.d + col0 + cc];
                }
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < cfg.d) {
                    vb = phi[(t + rr) * cfg.d + cfg.r_off + row0 + cc];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            var av: array<f32, 8>;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                av[i] = wg_a[kk * 64u + lid.x + 8u * i];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let bv = wg_b[kk * 64u + lid.y + 16u * j];
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    pb[j * 8u + i] = pb[j * 8u + i] + bv * av[i];
                }
            }
        }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y + 16u * j;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let c = col0 + lid.x + 8u * i;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[j * 8u + i];
                }
            }
        }
    }
}
