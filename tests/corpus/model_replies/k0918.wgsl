// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;

fn ld_b(t: u32, c: u32) -> f32 {
    if (t < cfg.m && c < cfg.d) { return phi[t * cfg.d + c]; }
    return 0.0;
}
fn ld_a(t: u32, lrow: u32) -> f32 {
    if (t < cfg.m && lrow < cfg.r_rows && (cfg.r_off + lrow) < cfg.d) {
        return phi[t * cfg.d + cfg.r_off + lrow];
    }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let it = lidx >> 4u;
    let g = lidx & 15u;
    let colb = wid.x * 64u + g * 4u;
    let rowa = wid.y * 64u + g * 4u;
    let slot = (it * 16u + g) & 255u;

    let nt = (cfg.m + 15u) / 16u;

    // preload tile 0 into buffer 0
    {
        let t = it;
        wg_b[slot] = vec4<f32>(ld_b(t, colb), ld_b(t, colb + 1u), ld_b(t, colb + 2u), ld_b(t, colb + 3u));
        wg_a[slot] = vec4<f32>(ld_a(t, rowa), ld_a(t, rowa + 1u), ld_a(t, rowa + 2u), ld_a(t, rowa + 3u));
    }
    workgroupBarrier();

    for (var i = 0u; i < nt; i = i + 1u) {
        let cur = (i & 1u) * 256u;
        let nxt = ((i + 1u) & 1u) * 256u;
        let t = (i + 1u) * 16u + it;
        let pb = vec4<f32>(ld_b(t, colb), ld_b(t, colb + 1u), ld_b(t, colb + 2u), ld_b(t, colb + 3u));
        let pa = vec4<f32>(ld_a(t, rowa), ld_a(t, rowa + 1u), ld_a(t, rowa + 2u), ld_a(t, rowa + 3u));

        for (var k = 0u; k < 16u; k = k + 1u) {
            let a = wg_a[(cur + k * 16u + lid.y) & 511u];
            let b = wg_b[(cur + k * 16u + lid.x) & 511u];
            acc0 = fma(vec4<f32>(a.x), b, acc0);
            acc1 = fma(vec4<f32>(a.y), b, acc1);
            acc2 = fma(vec4<f32>(a.z), b, acc2);
            acc3 = fma(vec4<f32>(a.w), b, acc3);
        }

        wg_b[(nxt + slot) & 511u] = pb;
        wg_a[(nxt + slot) & 511u] = pa;
        workgroupBarrier();
    }

    let c0 = wid.x * 64u + lid.x * 4u;
    let r0 = wid.y * 64u + lid.y * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = r0 + j;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            var v = acc0;
            if (j == 1u) { v = acc1; }
            if (j == 2u) { v = acc2; }
            if (j == 3u) { v = acc3; }
            let base = lr * cfg.d + c0;
            if (c0 < cfg.d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
