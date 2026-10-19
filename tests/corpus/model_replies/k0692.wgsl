// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 64>;
    var part: array<f32, 64>;
    var bv: array<f32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        acc[i] = 0.0;
        part[i] = 0.0;
        bv[i] = 0.0;
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 32u) {
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let e = i * 128u + lidx;
            let kk = e / 64u;
            let j = e % 64u;
            let t = t0 + kk;
            let c = col0 + j;
            var vb: f32 = 0.0;
            if (t < cfg.m && c < cfg.d) {
                vb = phi[t * cfg.d + c];
            }
            wg_b[e] = vb;
            let lr = row0 + j;
            let gr = cfg.r_off + lr;
            var va: f32 = 0.0;
            if (t < cfg.m && lr < cfg.r_rows && gr < cfg.d) {
                va = phi[t * cfg.d + gr];
            }
            wg_a[e] = va;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 32u; i = i + 1u) {
            part[i] = 0.0;
        }
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let kb = kk * 64u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                bv[q] = wg_b[kb + lid.x + 16u * q];
            }
            for (var p: u32 = 0u; p < 8u; p = p + 1u) {
                let a = wg_a[kb + lid.y + 8u * p];
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    part[p * 4u + q] = part[p * 4u + q] + a * bv[q];
                }
            }
        }
        for (var i: u32 = 0u; i < 32u; i = i + 1u) {
            acc[i] = acc[i] + part[i];
        }
        workgroupBarrier();
    }
    for (var p: u32 = 0u; p < 8u; p = p + 1u) {
        let lr = row0 + lid.y + 8u * p;
        let gr = cfg.r_off + lr;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let c = col0 + lid.x + 16u * q;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[p * 4u + q];
            }
        }
    }
}
