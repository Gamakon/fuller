// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

fn ld1(ok: bool, i: u32) -> f32 {
    var v: f32 = 0.0;
    if (ok) { v = phi[i]; }
    return v;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    let lidx = lid.y * 16u + lid.x;
    let litem = lidx >> 4u;
    let lquad = lidx & 15u;
    let cbase = wid.x * 64u + lquad * 4u;
    let rbase = wid.y * 64u + lquad * 4u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    var ra = vec4<f32>(0.0);
    var rb = vec4<f32>(0.0);
    // initial load
    if (litem < m) {
        let o = litem * d;
        rb = vec4<f32>(ld1(cbase < d, o + cbase), ld1(cbase + 1u < d, o + cbase + 1u),
                       ld1(cbase + 2u < d, o + cbase + 2u), ld1(cbase + 3u < d, o + cbase + 3u));
        ra = vec4<f32>(ld1(rbase < rr && ro + rbase < d, o + ro + rbase),
                       ld1(rbase + 1u < rr && ro + rbase + 1u < d, o + ro + rbase + 1u),
                       ld1(rbase + 2u < rr && ro + rbase + 2u < d, o + ro + rbase + 2u),
                       ld1(rbase + 3u < rr && ro + rbase + 3u < d, o + ro + rbase + 3u));
    }

    for (var t: u32 = 0u; t < m; t = t + 16u) {
        sa[lidx & 255u] = ra;
        sb[lidx & 255u] = rb;
        workgroupBarrier();
        // prefetch next tile
        ra = vec4<f32>(0.0);
        rb = vec4<f32>(0.0);
        let it = t + 16u + litem;
        if (it < m) {
            let o = it * d;
            rb = vec4<f32>(ld1(cbase < d, o + cbase), ld1(cbase + 1u < d, o + cbase + 1u),
                           ld1(cbase + 2u < d, o + cbase + 2u), ld1(cbase + 3u < d, o + cbase + 3u));
            ra = vec4<f32>(ld1(rbase < rr && ro + rbase < d, o + ro + rbase),
                           ld1(rbase + 1u < rr && ro + rbase + 1u < d, o + ro + rbase + 1u),
                           ld1(rbase + 2u < rr && ro + rbase + 2u < d, o + ro + rbase + 2u),
                           ld1(rbase + 3u < rr && ro + rbase + 3u < d, o + ro + rbase + 3u));
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sa[(k * 16u + lid.y) & 255u];
            let b = sb[(k * 16u + lid.x) & 255u];
            acc0 = fma(vec4<f32>(a.x), b, acc0);
            acc1 = fma(vec4<f32>(a.y), b, acc1);
            acc2 = fma(vec4<f32>(a.z), b, acc2);
            acc3 = fma(vec4<f32>(a.w), b, acc3);
        }
        workgroupBarrier();
    }

    let c0 = wid.x * 64u + lid.x * 4u;
    let r0 = wid.y * 64u + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < rr && ro + lr < d) {
            var av = acc0;
            if (i == 1u) { av = acc1; }
            if (i == 2u) { av = acc2; }
            if (i == 3u) { av = acc3; }
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + av.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + av.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + av.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + av.w; }
        }
    }
}
