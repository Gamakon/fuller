// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;

fn store_row(lr: u32, c0: u32, val: vec4<f32>) {
    let d = cfg.d;
    if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
        let base = lr * d + c0;
        if (c0 < d) { h[base] = h[base] + val.x; }
        if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + val.y; }
        if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + val.z; }
        if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + val.w; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let idx = lid.y * 16u + lid.x;
    let it = idx >> 4u;
    let v = idx & 15u;
    let colb = wid.x * 64u + v * 4u;
    let rowl = wid.y * 64u + v * 4u;
    let rowg = cfg.r_off + rowl;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        var vb = vec4<f32>(0.0);
        var va = vec4<f32>(0.0);
        let tt = t + it;
        if (tt < m) {
            let rb = tt * d;
            if (colb < d) { vb.x = phi[rb + colb]; }
            if (colb + 1u < d) { vb.y = phi[rb + colb + 1u]; }
            if (colb + 2u < d) { vb.z = phi[rb + colb + 2u]; }
            if (colb + 3u < d) { vb.w = phi[rb + colb + 3u]; }
            if (rowl < cfg.r_rows && rowg < d) { va.x = phi[rb + rowg]; }
            if (rowl + 1u < cfg.r_rows && rowg + 1u < d) { va.y = phi[rb + rowg + 1u]; }
            if (rowl + 2u < cfg.r_rows && rowg + 2u < d) { va.z = phi[rb + rowg + 2u]; }
            if (rowl + 3u < cfg.r_rows && rowg + 3u < d) { va.w = phi[rb + rowg + 3u]; }
        }
        wa[idx & 255u] = va;
        wb[idx & 255u] = vb;
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a = wa[(kk * 16u + lid.y) & 255u];
            let b = wb[(kk * 16u + lid.x) & 255u];
            a0 = a0 + a.x * b;
            a1 = a1 + a.y * b;
            a2 = a2 + a.z * b;
            a3 = a3 + a.w * b;
        }
        workgroupBarrier();
    }
    let lr0 = wid.y * 64u + lid.y * 4u;
    let c0 = wid.x * 64u + lid.x * 4u;
    store_row(lr0, c0, a0);
    store_row(lr0 + 1u, c0, a1);
    store_row(lr0 + 2u, c0, a2);
    store_row(lr0 + 3u, c0, a3);
}
