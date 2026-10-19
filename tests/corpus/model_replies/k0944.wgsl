// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

fn ld(i: u32, ok: bool) -> f32 {
    if (ok) { return phi[i]; }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lr0 = wid.y * 32u + lid.y;
    let lr1 = lr0 + 16u;
    let c0 = wid.x * 32u + lid.x;
    let c1 = c0 + 16u;
    let gr0 = cfg.r_off + lr0;
    let gr1 = cfg.r_off + lr1;
    let rok0 = (lr0 < cfg.r_rows) && (gr0 < d);
    let rok1 = (lr1 < cfg.r_rows) && (gr1 < d);
    let cok0 = c0 < d;
    let cok1 = c1 < d;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q & 15u] = 0.0; }

    var t: u32 = 0u;
    for (; t + 4u <= m; t = t + 4u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let base = (t + k) * d;
            let a0 = ld(base + gr0, rok0);
            let a1 = ld(base + gr1, rok1);
            let b0 = ld(base + c0, cok0);
            let b1 = ld(base + c1, cok1);
            let o = k * 4u;
            acc[(o) & 15u] = acc[(o) & 15u] + a0 * b0;
            acc[(o + 1u) & 15u] = acc[(o + 1u) & 15u] + a0 * b1;
            acc[(o + 2u) & 15u] = acc[(o + 2u) & 15u] + a1 * b0;
            acc[(o + 3u) & 15u] = acc[(o + 3u) & 15u] + a1 * b1;
        }
    }
    for (; t < m; t = t + 1u) {
        let base = t * d;
        let a0 = ld(base + gr0, rok0);
        let a1 = ld(base + gr1, rok1);
        let b0 = ld(base + c0, cok0);
        let b1 = ld(base + c1, cok1);
        acc[0u] = acc[0u] + a0 * b0;
        acc[1u] = acc[1u] + a0 * b1;
        acc[2u] = acc[2u] + a1 * b0;
        acc[3u] = acc[3u] + a1 * b1;
    }

    var s: array<f32, 4>;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        s[j & 3u] = (acc[j & 15u] + acc[(j + 4u) & 15u]) + (acc[(j + 8u) & 15u] + acc[(j + 12u) & 15u]);
    }

    if (rok0 && cok0) { let i = lr0 * d + c0; h[i] = h[i] + s[0u]; }
    if (rok0 && cok1) { let i = lr0 * d + c1; h[i] = h[i] + s[1u]; }
    if (rok1 && cok0) { let i = lr1 * d + c0; h[i] = h[i] + s[2u]; }
    if (rok1 && cok1) { let i = lr1 * d + c1; h[i] = h[i] + s[3u]; }
}
