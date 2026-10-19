// grid 32 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<f32, 16>;
    var cidx: array<u32, 16>;
    var ridx: array<u32, 16>;
    var cv: array<bool, 16>;
    var rv: array<bool, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
        cidx[k] = 0u;
        ridx[k] = 0u;
        cv[k] = false;
        rv[k] = false;
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let c = wid.x * 128u + j * 32u + lid.x;
        cv[j] = c < cfg.d;
        if (c < cfg.d) { cidx[j] = c; }
        let lr = wid.y * 32u + j * 8u + lid.y;
        let ok = lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d;
        rv[j] = ok;
        if (ok) { ridx[j] = cfg.r_off + lr; }
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let base = t * cfg.d;
        var a: array<f32, 4>;
        var b: array<f32, 4>;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            var x: f32 = 0.0;
            var y: f32 = 0.0;
            if (rv[j]) { x = phi[base + ridx[j]]; }
            if (cv[j]) { y = phi[base + cidx[j]]; }
            a[j] = x;
            b[j] = y;
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            if (rv[i] && cv[j]) {
                let lr = wid.y * 32u + i * 8u + lid.y;
                let c = wid.x * 128u + j * 32u + lid.x;
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[i * 4u + j];
            }
        }
    }
}
