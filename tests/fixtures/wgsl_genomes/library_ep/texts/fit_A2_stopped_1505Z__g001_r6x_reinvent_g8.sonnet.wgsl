// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn ld3(a: bool, b: bool, c: bool, v: f32) -> f32 {
    return select(0.0, select(0.0, select(0.0, v, c), b), a);
}
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    var pw: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
        pw[k] = vec4<f32>(0.0);
    }
    let i7a = lidx;
    let i7b = lidx + 128u;
    let i9a = (i7a & 15u) * 4u;
    let i9b = (i7b & 15u) * 4u;
    let cba = wid.x * 64u + i9a;
    let rba = wid.y * 64u + i9a;
    let cbb = wid.x * 64u + i9b;
    let rbb = wid.y * 64u + i9b;
    let gra = cfg.r_off + rba;
    let grb = cfg.r_off + rbb;
    for (var t: u32 = 0u; t < cfg.m; t = t + 64u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            pw[k] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i5 = t + k * 16u;
            let i8a = i5 + (i7a >> 4u);
            let i8b = i5 + (i7b >> 4u);
            let i10a = i8a * cfg.d + cba;
            let j10a = i8a * cfg.d + gra;
            let i10b = i8b * cfg.d + cbb;
            let j10b = i8b * cfg.d + grb;
            let oka = i8a < cfg.m;
            let okb = i8b < cfg.m;
            let va = vec4<f32>(
                ld3(oka, cba < cfg.d, true, phi[i10a]),
                ld3(oka, cba + 1u < cfg.d, true, phi[i10a + 1u]),
                ld3(oka, cba + 2u < cfg.d, true, phi[i10a + 2u]),
                ld3(oka, cba + 3u < cfg.d, true, phi[i10a + 3u]));
            let wa = vec4<f32>(
                ld3(oka, rba < cfg.r_rows, gra < cfg.d, phi[j10a]),
                ld3(oka, rba + 1u < cfg.r_rows, gra + 1u < cfg.d, phi[j10a + 1u]),
                ld3(oka, rba + 2u < cfg.r_rows, gra + 2u < cfg.d, phi[j10a + 2u]),
                ld3(oka, rba + 3u < cfg.r_rows, gra + 3u < cfg.d, phi[j10a + 3u]));
            let vb = vec4<f32>(
                ld3(okb, cbb < cfg.d, true, phi[i10b]),
                ld3(okb, cbb + 1u < cfg.d, true, phi[i10b + 1u]),
                ld3(okb, cbb + 2u < cfg.d, true, phi[i10b + 2u]),
                ld3(okb, cbb + 3u < cfg.d, true, phi[i10b + 3u]));
            let wb = vec4<f32>(
                ld3(okb, rbb < cfg.r_rows, grb < cfg.d, phi[j10b]),
                ld3(okb, rbb + 1u < cfg.r_rows, grb + 1u < cfg.d, phi[j10b + 1u]),
                ld3(okb, rbb + 2u < cfg.r_rows, grb + 2u < cfg.d, phi[j10b + 2u]),
                ld3(okb, rbb + 3u < cfg.r_rows, grb + 3u < cfg.d, phi[j10b + 3u]));
            wg_va[i7a] = va;
            wg_vb[i7a] = wa;
            wg_va[i7b] = vb;
            wg_vb[i7b] = wb;
            workgroupBarrier();
            for (var k3: u32 = 0u; k3 < 8u; k3 = k3 + 1u) {
                pw[k3 + 8u] = vec4<f32>(0.0);
            }
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let a0 = wg_va[q * 16u + lid.x];
                let a1 = wg_va[q * 16u + lid.x + 8u];
                let b = wg_vb[q * 16u + lid.y];
                pw[8] = pw[8] + b.x * a0;
                pw[9] = pw[9] + b.x * a1;
                pw[10] = pw[10] + b.y * a0;
                pw[11] = pw[11] + b.y * a1;
                pw[12] = pw[12] + b.z * a0;
                pw[13] = pw[13] + b.z * a1;
                pw[14] = pw[14] + b.w * a0;
                pw[15] = pw[15] + b.w * a1;
            }
            for (var k4: u32 = 0u; k4 < 8u; k4 = k4 + 1u) {
                pw[k4] = pw[k4] + pw[k4 + 8u];
            }
            workgroupBarrier();
        }
        for (var k5: u32 = 0u; k5 < 8u; k5 = k5 + 1u) {
            pv[k5] = pv[k5] + pw[k5];
        }
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + k;
        let acc0 = pv[k * 2u];
        let acc1 = pv[k * 2u + 1u];
        let c0 = wid.x * 64u + lid.x * 4u;
        let c1 = c0 + 32u;
        let base = lr * cfg.d;
        if (lr < cfg.r_rows) {
            if (cfg.r_off + lr < cfg.d) {
                if (c0 < cfg.d) { h[base + c0] = h[base + c0] + acc0.x; }
                if (c0 + 1u < cfg.d) { h[base + c0 + 1u] = h[base + c0 + 1u] + acc0.y; }
                if (c0 + 2u < cfg.d) { h[base + c0 + 2u] = h[base + c0 + 2u] + acc0.z; }
                if (c0 + 3u < cfg.d) { h[base + c0 + 3u] = h[base + c0 + 3u] + acc0.w; }
                if (c1 < cfg.d) { h[base + c1] = h[base + c1] + acc1.x; }
                if (c1 + 1u < cfg.d) { h[base + c1 + 1u] = h[base + c1 + 1u] + acc1.y; }
                if (c1 + 2u < cfg.d) { h[base + c1 + 2u] = h[base + c1 + 2u] + acc1.z; }
                if (c1 + 3u < cfg.d) { h[base + c1 + 3u] = h[base + c1 + 3u] + acc1.w; }
            }
        }
    }
}
