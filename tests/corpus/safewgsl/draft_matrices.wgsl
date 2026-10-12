// grid 16 16 1 1
// Matrices (2026-10-12): composed from scalars and from 3 and 4 columns, products with vectors, scalars and matrices, sums, negation, columns by literal and at run time, a matrix local with a column store.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var g: f32 = 0.0;
    var n: u32 = gid.x;
    var v: vec4<f32> = vec4<f32>(0.5);
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
    }
    let m2 = mat2x2<f32>(f0, 1.0, 0.0, f0);
    let m3 = mat3x3<f32>(vec3<f32>(f0), vec3<f32>(1.0), vec3<f32>(0.0, 0.0, 1.0));
    let m34 = mat3x4<f32>(v, v, vec4<f32>(f0));
    let m4 = mat4x4<f32>(v, v, v, vec4<f32>(f0));
    var mv = m2;
    mv[1] = vec2<f32>(2.0, f0);
    g = determinant(m2) + (m2 * vec2<f32>(1.0)).x + (vec2<f32>(1.0) * m2).y + determinant(m3) + (m34 * vec3<f32>(1.0)).w;
    g = g + (m4 * v).x + (transpose(m2) * m2)[1][0] + (m2 * 2.0 + m2 - m2)[0].x + (-m2)[1].y + mv[1].y + mv[0][0] + mv[n & 1u].x;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
