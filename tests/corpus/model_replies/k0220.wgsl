// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;

fn load4(item: u32, col: u32, lim_rows: u32, base_off: u32) -> vec4<f32> {
    // loads phi[item*d + base_off + col + j] for j in 0..3, guarded by col+j < lim_rows and base_off+col+j < d
    var r = vec4<f32>(0.0);
    if (item < cfg.m) {
        let base = item * cfg.d + base_off + col;
        if (col < lim_rows && base_off + col < cfg.d) { r.x = phi[base]; }
        if (col + 1u < lim_rows && base_off + col + 1u < cfg.d) { r.y = phi[base + 1u]; }
        if (col + 2u < lim_rows && base_off + col + 2u < cfg.d) { r.z = phi[base + 2u]; }
        if (col + 3u < lim_rows && base_off + col + 3u < cfg.d) { r.w = phi[base + 3u]; }
    }
    return r;
}

fn put(row: u32, col: u32, v: vec4<f32>) {
    if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
        let b = row * cfg.d + col;
        if (col < cfg.d) { h[b] = h[b] + v.x; }
        if (col + 1u < cfg.d) { h[b + 1u] = h[b + 1u] + v.y; }
        if (col + 2u < cfg.d) { h[b + 2u] = h[b + 2u] + v.z; }
        if (col + 3u < cfg.d) { h[b + 3u] = h[b + 3u] + v.w; }
    }
}

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let off = (idx & 15u) * 4u;
            wg_va[idx & 511u] = load4(item, col0 + off, cfg.d, 0u);
            wg_vb[idx & 511u] = load4(item, row0 + off, cfg.r_rows, cfg.r_off);
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b0 = wg_vb[(k * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(k * 16u + lid.y * 2u + 1u) & 511u];
            a0 = a0 + vec4<f32>(b0.x) * a;
            a1 = a1 + vec4<f32>(b0.y) * a;
            a2 = a2 + vec4<f32>(b0.z) * a;
            a3 = a3 + vec4<f32>(b0.w) * a;
            a4 = a4 + vec4<f32>(b1.x) * a;
            a5 = a5 + vec4<f32>(b1.y) * a;
            a6 = a6 + vec4<f32>(b1.z) * a;
            a7 = a7 + vec4<f32>(b1.w) * a;
        }
        workgroupBarrier();
    }
    let col = col0 + lid.x * 4u;
    let row = row0 + lid.y * 8u;
    put(row, col, a0);
    put(row + 1u, col, a1);
    put(row + 2u, col, a2);
    put(row + 3u, col, a3);
    put(row + 4u, col, a4);
    put(row + 5u, col, a5);
    put(row + 6u, col, a6);
    put(row + 7u, col, a7);
}
