// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 32>;
    var tmp: array<f32, 32>;
    for (var q: u32 = 0u; q < 32u; q = q + 1u) {
        acc[q] = 0.0;
        tmp[q] = 0.0;
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 128u;
    let nt = (cfg.m + 15u) / 16u;
    for (var tb: u32 = 0u; tb < nt; tb = tb + 1u) {
        let t = tb * 16u;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e >> 6u;
            let c = col0 + (e & 63u);
            var v: f32 = 0.0;
            if (t + kk < cfg.m && c < cfg.d) {
                v = phi[(t + kk) * cfg.d + c];
            }
            wg_a[e] = v;
        }
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e >> 7u;
            let lr = row0 + (e & 127u);
            var v: f32 = 0.0;
            if (t + kk < cfg.m && lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                v = phi[(t + kk) * cfg.d + cfg.r_off + lr];
            }
            wg_b[e] = v;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            var a: array<f32, 4>;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                a[j] = wg_a[kk * 64u + lid.x + 16u * j];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let b = wg_b[kk * 128u + lid.y + 16u * i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    tmp[i * 4u + j] = tmp[i * 4u + j] + b * a[j];
                }
            }
        }
        for (var q: u32 = 0u; q < 32u; q = q + 1u) {
            acc[q] = acc[q] + tmp[q];
            tmp[q] = 0.0;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
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
