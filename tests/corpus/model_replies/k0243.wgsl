// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 2048>;
var<workgroup> wb: array<f32, 2048>;

fn store_row(lr: u32, c: u32, v: vec4<f32>) {
    if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
        let base = lr * cfg.d + c;
        if (c < cfg.d) { h[base] = h[base] + v.x; }
        if (c + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
        if (c + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
        if (c + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
    }
}

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let cx = lid.x * 4u;
    let ry = lid.y * 8u;
    let m = cfg.m;
    let d = cfg.d;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let e = lidx + 128u * k;
            let it = e >> 6u;
            let cl = e & 63u;
            let ti = t + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (ti < m) {
                let base = ti * d;
                let c = colbase + cl;
                if (c < d) { va = phi[base + c]; }
                let lr = rowbase + cl;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    vb = phi[base + cfg.r_off + lr];
                }
            }
            wa[e & 2047u] = va;
            wb[e & 2047u] = vb;
        }
        workgroupBarrier();
        for (var q: u32 = 0u; q < 32u; q = q + 1u) {
            let o = q * 64u;
            let a = vec4<f32>(
                wa[(o + cx) & 2047u],
                wa[(o + cx + 1u) & 2047u],
                wa[(o + cx + 2u) & 2047u],
                wa[(o + cx + 3u) & 2047u]);
            a0 = a0 + wb[(o + ry) & 2047u] * a;
            a1 = a1 + wb[(o + ry + 1u) & 2047u] * a;
            a2 = a2 + wb[(o + ry + 2u) & 2047u] * a;
            a3 = a3 + wb[(o + ry + 3u) & 2047u] * a;
            a4 = a4 + wb[(o + ry + 4u) & 2047u] * a;
            a5 = a5 + wb[(o + ry + 5u) & 2047u] * a;
            a6 = a6 + wb[(o + ry + 6u) & 2047u] * a;
            a7 = a7 + wb[(o + ry + 7u) & 2047u] * a;
        }
        workgroupBarrier();
    }
    let c0 = colbase + cx;
    let r0 = rowbase + ry;
    store_row(r0, c0, a0);
    store_row(r0 + 1u, c0, a1);
    store_row(r0 + 2u, c0, a2);
    store_row(r0 + 3u, c0, a3);
    store_row(r0 + 4u, c0, a4);
    store_row(r0 + 5u, c0, a5);
    store_row(r0 + 6u, c0, a6);
    store_row(r0 + 7u, c0, a7);
}
