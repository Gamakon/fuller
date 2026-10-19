// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;

fn load4(item: u32, start: u32, limit: u32) -> vec4<f32> {
    var r = vec4<f32>(0.0);
    if (item < cfg.m) {
        let base = item * cfg.d;
        if (start < limit) { r.x = phi[base + start]; }
        if (start + 1u < limit) { r.y = phi[base + start + 1u]; }
        if (start + 2u < limit) { r.z = phi[base + start + 2u]; }
        if (start + 3u < limit) { r.w = phi[base + start + 3u]; }
    }
    return r;
}

fn store_row(lr: u32, c0: u32, v: vec4<f32>) {
    if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
        let base = lr * cfg.d + c0;
        if (c0 < cfg.d) { h[base] = h[base] + v.x; }
        if (c0 + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
        if (c0 + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
        if (c0 + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let ir = lidx >> 4u;
    let cg = (lidx & 15u) * 4u;
    let bstart = wid.x * 64u + cg;
    let astart = cfg.r_off + wid.y * 64u + cg;
    let alimit = min(cfg.d, cfg.r_off + cfg.r_rows);
    let blimit = cfg.d;
    let sidx = lidx & 255u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    wg_a[sidx] = load4(ir, astart, alimit);
    wg_b[sidx] = load4(ir, bstart, blimit);
    workgroupBarrier();

    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        let cur = ((t >> 4u) & 1u) * 256u;
        let nxt = 256u - cur;
        let na = load4(t + 16u + ir, astart, alimit);
        let nb = load4(t + 16u + ir, bstart, blimit);
        wg_a[(nxt + sidx) & 511u] = na;
        wg_b[(nxt + sidx) & 511u] = nb;
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let av = wg_a[(cur + kk * 16u + lid.y) & 511u];
            let bv = wg_b[(cur + kk * 16u + lid.x) & 511u];
            acc0 = fma(vec4<f32>(av.x), bv, acc0);
            acc1 = fma(vec4<f32>(av.y), bv, acc1);
            acc2 = fma(vec4<f32>(av.z), bv, acc2);
            acc3 = fma(vec4<f32>(av.w), bv, acc3);
        }
        workgroupBarrier();
    }

    let r0 = wid.y * 64u + lid.y * 4u;
    let c0 = wid.x * 64u + lid.x * 4u;
    store_row(r0, c0, acc0);
    store_row(r0 + 1u, c0, acc1);
    store_row(r0 + 2u, c0, acc2);
    store_row(r0 + 3u, c0, acc3);
}
