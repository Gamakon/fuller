// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

fn prod(base: u32, r0: u32, r1: u32, c0: u32, c1: u32) -> vec4<f32> {
    let rv = vec2<f32>(phi[base + r0], phi[base + r1]);
    let cv = vec2<f32>(phi[base + c0], phi[base + c1]);
    return vec4<f32>(rv.x * cv, rv.y * cv);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let cb = wg.x * 32u + lid.x;
    let rb = wg.y * 32u + lid.y;
    let c0 = cb;
    let c1 = cb + 16u;
    let l0 = rb;
    let l1 = rb + 16u;
    let g0 = cfg.r_off + l0;
    let g1 = cfg.r_off + l1;
    let v_c0 = c0 < d;
    let v_c1 = c1 < d;
    let v_r0 = (l0 < cfg.r_rows) && (g0 < d);
    let v_r1 = (l1 < cfg.r_rows) && (g1 < d);
    let dm = max(d, 1u) - 1u;
    let cc0 = min(c0, dm);
    let cc1 = min(c1, dm);
    let rr0 = min(g0, dm);
    let rr1 = min(g1, dm);

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t: u32 = 0u;
    let m4 = m & ~3u;
    loop {
        if (t >= m4) { break; }
        let b0 = t * d;
        a0 = a0 + prod(b0, rr0, rr1, cc0, cc1);
        a1 = a1 + prod(b0 + d, rr0, rr1, cc0, cc1);
        a2 = a2 + prod(b0 + 2u * d, rr0, rr1, cc0, cc1);
        a3 = a3 + prod(b0 + 3u * d, rr0, rr1, cc0, cc1);
        t = t + 4u;
    }
    loop {
        if (t >= m) { break; }
        a0 = a0 + prod(t * d, rr0, rr1, cc0, cc1);
        t = t + 1u;
    }
    let s = (a0 + a1) + (a2 + a3);

    if (v_r0 && v_c0) {
        let idx = l0 * d + c0;
        h[idx] = h[idx] + s.x;
    }
    if (v_r0 && v_c1) {
        let idx = l0 * d + c1;
        h[idx] = h[idx] + s.y;
    }
    if (v_r1 && v_c0) {
        let idx = l1 * d + c0;
        h[idx] = h[idx] + s.z;
    }
    if (v_r1 && v_c1) {
        let idx = l1 * d + c1;
        h[idx] = h[idx] + s.w;
    }
}
