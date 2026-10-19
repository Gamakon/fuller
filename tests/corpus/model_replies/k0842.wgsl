// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wg.x * 32u + lid.x * 4u;
    let row0 = wg.y * 32u + lid.y * 4u;

    var rv: array<bool, 4>;
    var cv: array<bool, 4>;
    var rg: array<u32, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i;
        let gr = cfg.r_off + lr;
        rv[i] = (lr < cfg.r_rows) && (gr < d);
        rg[i] = gr;
        cv[i] = (col0 + i) < d;
    }

    var acc: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }

    var t: u32 = 0u;
    loop {
        if (t + 4u > m) { break; }
        for (var u: u32 = 0u; u < 4u; u = u + 1u) {
            let base = (t + u) * d;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i] = select(0.0, phi[base + rg[i]], rv[i]);
                b[i] = select(0.0, phi[base + col0 + i], cv[i]);
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        t = t + 4u;
    }
    loop {
        if (t >= m) { break; }
        let base = t * d;
        var a: array<f32, 4>;
        var b: array<f32, 4>;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            a[i] = select(0.0, phi[base + rg[i]], rv[i]);
            b[i] = select(0.0, phi[base + col0 + i], cv[i]);
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
            }
        }
        t = t + 1u;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}