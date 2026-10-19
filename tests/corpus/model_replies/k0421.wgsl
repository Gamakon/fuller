// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<f32, 32>;
    var ra: array<f32, 8>;
    var cb: array<f32, 4>;
    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 8u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let lr = row0 + i;
            let gr = cfg.r_off + lr;
            var a: f32 = 0.0;
            if (lr < cfg.r_rows && gr < cfg.d) {
                a = phi[base + gr];
            }
            ra[i] = a;
        }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + j;
            var b: f32 = 0.0;
            if (c < cfg.d) {
                b = phi[base + c];
            }
            cb[j] = b;
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                acc[i * 4u + j] = acc[i * 4u + j] + ra[i] * cb[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + i;
        let gr = cfg.r_off + lr;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[i * 4u + j];
            }
        }
    }
}
