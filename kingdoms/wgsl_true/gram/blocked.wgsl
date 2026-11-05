// grid 16 16 4 4
// A register-blocked Gram without workgroup memory: each thread owns a
// 4 x 4 block of H and, per item, loads its 4 row values and 4 column
// values straight from phi (two 16-byte loads, coalesced across the
// workgroup) for 16 multiply-adds. No barriers. Rows band-local as shipped.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
fn load4(t: u32, base: u32, bound: u32) -> vec4<f32> {
    let p = t * cfg.d + base;
    var v = vec4<f32>(0.0);
    if (base + 3u < bound) {
        v = vec4<f32>(phi[p], phi[p + 1u], phi[p + 2u], phi[p + 3u]);
    } else {
        if (base < bound) { v.x = phi[p]; }
        if (base + 1u < bound) { v.y = phi[p + 1u]; }
        if (base + 2u < bound) { v.z = phi[p + 2u]; }
    }
    return v;
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lr0 = gid.y * 4u;
    let c0 = gid.x * 4u;
    if (lr0 >= cfg.r_rows || c0 >= cfg.d) { return; }
    let row_bound = min(cfg.r_rows, cfg.d - min(cfg.r_off, cfg.d));
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        let rv = load4(t, cfg.r_off + lr0, cfg.r_off + row_bound);
        let cv = load4(t, c0, cfg.d);
        acc0 = acc0 + rv.x * cv;
        acc1 = acc1 + rv.y * cv;
        acc2 = acc2 + rv.z * cv;
        acc3 = acc3 + rv.w * cv;
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = lr0 + i;
        if (lr >= cfg.r_rows) { continue; }
        var a = acc0;
        if (i == 1u) { a = acc1; }
        if (i == 2u) { a = acc2; }
        if (i == 3u) { a = acc3; }
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = c0 + j;
            if (c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + a[j];
            }
        }
    }
}
