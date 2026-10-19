// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    var rows: array<u32, 4>;
    var cols: array<u32, 4>;
    var rv: array<bool, 4>;
    var cv: array<bool, 4>;
    var lrs: array<u32, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 32u + i * 8u + lid.y;
        let gr = cfg.r_off + lr;
        let ok = lr < cfg.r_rows && gr < d;
        rv[i] = ok;
        lrs[i] = lr;
        rows[i] = select(0u, gr, ok);
        let c = wid.x * 32u + i * 8u + lid.x;
        let okc = c < d;
        cv[i] = okc;
        cols[i] = select(0u, c, okc);
    }
    var acc: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { acc[k] = 0.0; }

    var t: u32 = 0u;
    let m4 = m & ~3u;
    for (; t < m4; t = t + 4u) {
        for (var u: u32 = 0u; u < 4u; u = u + 1u) {
            let base = (t + u) * d;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i] = phi[base + rows[i]];
                b[i] = phi[base + cols[i]];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
    }
    for (; t < m; t = t + 1u) {
        let base = t * d;
        var a: array<f32, 4>;
        var b: array<f32, 4>;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            a[i] = phi[base + rows[i]];
            b[i] = phi[base + cols[i]];
        }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
            }
        }
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if (rv[i]) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                if (cv[j]) {
                    let idx = lrs[i] * d + cols[j];
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}