// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 8u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            var a: f32 = 0.0;
            if (col0 + j < cfg.d) { a = phi[base + col0 + j]; }
            pb[j] = a;
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            var b: f32 = 0.0;
            if (row0 + i < cfg.r_rows && cfg.r_off + row0 + i < cfg.d) { b = phi[base + cfg.r_off + row0 + i]; }
            pc[i] = b;
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                pa[i * 4u + j] = pa[i * 4u + j] + pc[i] * pb[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            if (row0 + i < cfg.r_rows && col0 + j < cfg.d) {
                let idx = (row0 + i) * cfg.d + col0 + j;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
