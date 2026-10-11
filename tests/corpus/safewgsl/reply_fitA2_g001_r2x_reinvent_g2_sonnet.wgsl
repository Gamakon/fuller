// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var tot: array<vec4<f32>, 4>;
    tot[0] = vec4<f32>(0.0);
    tot[1] = vec4<f32>(0.0);
    tot[2] = vec4<f32>(0.0);
    tot[3] = vec4<f32>(0.0);
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let row = idx / 64u;
            let col = idx % 64u;
            let item = t + row;
            let oka = item < cfg.m && colbase + col < cfg.d;
            let okb = item < cfg.m && rowbase + col < cfg.r_rows && cfg.r_off + rowbase + col < cfg.d;
            let ia = (item * cfg.d + colbase + col) * u32(oka);
            let ib = (item * cfg.d + cfg.r_off + rowbase + col) * u32(okb);
            let va = phi[ia] * f32(oka);
            let vb = phi[ib] * f32(okb);
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        var acc: array<vec4<f32>, 4>;
        acc[0] = vec4<f32>(0.0);
        acc[1] = vec4<f32>(0.0);
        acc[2] = vec4<f32>(0.0);
        acc[3] = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let a0 = wg_a[kk * 64u + lid.x];
            let a1 = wg_a[kk * 64u + lid.x + 32u];
            let av = vec4<f32>(a0, a1, a0, a1);
            let bo = kk * 64u + lid.y * 8u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let b0 = wg_b[bo + 2u * q];
                let b1 = wg_b[bo + 2u * q + 1u];
                acc[q] = acc[q] + vec4<f32>(b0, b0, b1, b1) * av;
            }
        }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            tot[q] = tot[q] + acc[q];
        }
        workgroupBarrier();
    }
    let c0 = colbase + lid.x;
    let c1 = c0 + 32u;
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        let v = tot[q];
        let lr0 = rowbase + lid.y * 8u + 2u * q;
        let lr1 = lr0 + 1u;
        let r0ok = lr0 < cfg.r_rows && cfg.r_off + lr0 < cfg.d;
        let r1ok = lr1 < cfg.r_rows && cfg.r_off + lr1 < cfg.d;
        let w00 = r0ok && c0 < cfg.d;
        let w01 = r0ok && c1 < cfg.d;
        let w10 = r1ok && c0 < cfg.d;
        let w11 = r1ok && c1 < cfg.d;
        if (w00) {
            h[lr0 * cfg.d + c0] = h[lr0 * cfg.d + c0] + v.x;
        }
        if (w01) {
            h[lr0 * cfg.d + c1] = h[lr0 * cfg.d + c1] + v.y;
        }
        if (w10) {
            h[lr1 * cfg.d + c0] = h[lr1 * cfg.d + c0] + v.z;
        }
        if (w11) {
            h[lr1 * cfg.d + c1] = h[lr1 * cfg.d + c1] + v.w;
        }
    }
}
