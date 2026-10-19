// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 8, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 8u;
    var acc: array<f32, 64>;
    var rv: array<f32, 64>;
    var cv: array<f32, 64>;
    var r: u32;
    var c: u32;
    var t: u32;
    var q: u32;

    for (r = 0u; r < 32u; r = r + 1u) {
        acc[r] = 0.0;
    }

    for (t = 0u; t < cfg.m; t = t + 4u) {
        for (q = 0u; q < 4u; q = q + 1u) {
            if (t + q < cfg.m) {
                for (r = 0u; r < 8u; r = r + 1u) {
                    let lr = row0 + r;
                    let gr = cfg.r_off + lr;
                    if (lr < cfg.r_rows && gr < cfg.d) {
                        rv[r] = phi[(t + q) * cfg.d + gr];
                    } else {
                        rv[r] = 0.0;
                    }
                }
                for (c = 0u; c < 4u; c = c + 1u) {
                    let cc = col0 + c;
                    if (cc < cfg.d) {
                        cv[c] = phi[(t + q) * cfg.d + cc];
                    } else {
                        cv[c] = 0.0;
                    }
                }
                for (r = 0u; r < 8u; r = r + 1u) {
                    for (c = 0u; c < 4u; c = c + 1u) {
                        acc[r * 4u + c] = acc[r * 4u + c] + rv[r] * cv[c];
                    }
                }
            }
        }
    }

    for (r = 0u; r < 8u; r = r + 1u) {
        let lr = row0 + r;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < cfg.d) {
            for (c = 0u; c < 4u; c = c + 1u) {
                let cc = col0 + c;
                if (cc < cfg.d) {
                    let out = lr * cfg.d + cc;
                    h[out] = h[out] + acc[r * 4u + c];
                }
            }
        }
    }
}
