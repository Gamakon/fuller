// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sc: array<vec4<f32>, 256>;
var<workgroup> sr: array<vec4<f32>, 256>;

fn ldv(tt: u32, c0: u32, lim: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (tt < cfg.m) {
        let base = tt * cfg.d;
        if (c0 < lim) { v.x = phi[base + c0]; }
        if (c0 + 1u < lim) { v.y = phi[base + c0 + 1u]; }
        if (c0 + 2u < lim) { v.z = phi[base + c0 + 2u]; }
        if (c0 + 3u < lim) { v.w = phi[base + c0 + 3u]; }
    }
    return v;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let li = ty * 16u + tx;
    let cx0 = wid.x * 64u;
    let ry0 = wid.y * 64u;
    let lt = li >> 4u;
    let lj = (li & 15u) * 4u;
    // column vec4 start for loader
    let ccol = cx0 + lj;
    // row limit: local row must be < r_rows and global row < d
    let rlim = min(cfg.r_rows, select(0u, cfg.d - cfg.r_off, cfg.d > cfg.r_off));
    let rloc = ry0 + lj;
    let rglob = cfg.r_off + rloc;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        let tt = t0 + lt;
        sc[li] = ldv(tt, ccol, cfg.d);
        // rows: valid count limited by rlim (local index space)
        var rv = vec4<f32>(0.0);
        if (tt < cfg.m) {
            let base = tt * cfg.d;
            if (rloc < rlim) { rv.x = phi[base + rglob]; }
            if (rloc + 1u < rlim) { rv.y = phi[base + rglob + 1u]; }
            if (rloc + 2u < rlim) { rv.z = phi[base + rglob + 2u]; }
            if (rloc + 3u < rlim) { rv.w = phi[base + rglob + 3u]; }
        }
        sr[li] = rv;
        workgroupBarrier();
        for (var t: u32 = 0u; t < 16u; t = t + 1u) {
            let cv = sc[t * 16u + tx];
            let r = sr[t * 16u + ty];
            a0 = a0 + r.x * cv;
            a1 = a1 + r.y * cv;
            a2 = a2 + r.z * cv;
            a3 = a3 + r.w * cv;
        }
        workgroupBarrier();
    }

    let c0 = cx0 + tx * 4u;
    let lr0 = ry0 + ty * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = lr0 + i;
        if (lr < rlim) {
            var a = a0;
            if (i == 1u) { a = a1; }
            if (i == 2u) { a = a2; }
            if (i == 3u) { a = a3; }
            let o = lr * cfg.d + c0;
            if (c0 < cfg.d) { h[o] = h[o] + a.x; }
            if (c0 + 1u < cfg.d) { h[o + 1u] = h[o + 1u] + a.y; }
            if (c0 + 2u < cfg.d) { h[o + 2u] = h[o + 2u] + a.z; }
            if (c0 + 3u < cfg.d) { h[o + 3u] = h[o + 3u] + a.w; }
        }
    }
}
