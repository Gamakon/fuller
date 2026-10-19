// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let c0 = wg.x * 32u + li.x;
    let c1 = c0 + 16u;
    let l0 = wg.y * 32u + li.y;
    let l1 = l0 + 16u;
    let g0 = cfg.r_off + l0;
    let g1 = cfg.r_off + l1;
    let dm = d - 1u;
    let cc0 = min(c0, dm);
    let cc1 = min(c1, dm);
    let rr0 = min(g0, dm);
    let rr1 = min(g1, dm);

    var a00 = 0.0; var a01 = 0.0; var a10 = 0.0; var a11 = 0.0;
    var b00 = 0.0; var b01 = 0.0; var b10 = 0.0; var b11 = 0.0;

    var t: u32 = 0u;
    var base: u32 = 0u;
    let m2 = m & ~1u;
    for (; t < m2; t = t + 2u) {
        let p0 = base;
        let p1 = base + d;
        let x0 = phi[p0 + rr0];
        let x1 = phi[p0 + rr1];
        let y0 = phi[p0 + cc0];
        let y1 = phi[p0 + cc1];
        let u0 = phi[p1 + rr0];
        let u1 = phi[p1 + rr1];
        let v0 = phi[p1 + cc0];
        let v1 = phi[p1 + cc1];
        a00 = fma(x0, y0, a00);
        a01 = fma(x0, y1, a01);
        a10 = fma(x1, y0, a10);
        a11 = fma(x1, y1, a11);
        b00 = fma(u0, v0, b00);
        b01 = fma(u0, v1, b01);
        b10 = fma(u1, v0, b10);
        b11 = fma(u1, v1, b11);
        base = base + 2u * d;
    }
    if (t < m) {
        let x0 = phi[base + rr0];
        let x1 = phi[base + rr1];
        let y0 = phi[base + cc0];
        let y1 = phi[base + cc1];
        a00 = fma(x0, y0, a00);
        a01 = fma(x0, y1, a01);
        a10 = fma(x1, y0, a10);
        a11 = fma(x1, y1, a11);
    }

    let ok0 = l0 < cfg.r_rows && g0 < d;
    let ok1 = l1 < cfg.r_rows && g1 < d;
    let cv0 = c0 < d;
    let cv1 = c1 < d;
    if (ok0 && cv0) {
        let i = l0 * d + c0;
        h[i] = h[i] + (a00 + b00);
    }
    if (ok0 && cv1) {
        let i = l0 * d + c1;
        h[i] = h[i] + (a01 + b01);
    }
    if (ok1 && cv0) {
        let i = l1 * d + c0;
        h[i] = h[i] + (a10 + b10);
    }
    if (ok1 && cv1) {
        let i = l1 * d + c1;
        h[i] = h[i] + (a11 + b11);
    }
}
