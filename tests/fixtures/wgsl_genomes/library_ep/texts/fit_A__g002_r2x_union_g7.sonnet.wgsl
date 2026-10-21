// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u;
    var acc: array<f32, 32>;
    var pa: array<f32, 4>;
    var pb: array<f32, 4>;
    for (var q: u32 = 0u; q < 32u; q = q + 1u) {
    acc[q] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    i0 = ((lid.y * 16u) + lid.x) + (k * 256u);
    i1 = i0 >> 6u;
    i2 = i0 & 63u;
    f0 = 0.0;
    f1 = 0.0;
    if (((t + i1) < cfg.m) && (((wid.x * 64u) + i2) < cfg.d)) {
    f0 = phi[((t + i1) * cfg.d) + ((wid.x * 64u) + i2)];
    }
    if ((((t + i1) < cfg.m) && (((wid.y * 64u) + i2) < cfg.r_rows)) && ((cfg.r_off + ((wid.y * 64u) + i2)) < cfg.d)) {
    f1 = phi[(((t + i1) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + i2)];
    }
    wg_a[i0 & 2047u] = f0;
    wg_b[i0 & 2047u] = f1;
    }
    workgroupBarrier();
    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    i3 = (((g * 8u) + j) * 64u) + (lid.x * 4u);
    i4 = (((g * 8u) + j) * 64u) + (lid.y * 4u);
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    pa[c] = wg_a[(i3 + c) & 2047u];
    pb[c] = wg_b[(i4 + c) & 2047u];
    }
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    acc[(g * 16u) + (r * 4u) + c] = acc[(g * 16u) + (r * 4u) + c] + (pa[c] * pb[r]);
    }
    }
    }
    }
    workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
    i0 = (gid.y * 4u) + r;
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    i1 = (gid.x * 4u) + c;
    if (((i0 < cfg.r_rows) && ((cfg.r_off + i0) < cfg.d)) && (i1 < cfg.d)) {
    h[(i0 * cfg.d) + i1] = h[(i0 * cfg.d) + i1] + (acc[(r * 4u) + c] + acc[16u + (r * 4u) + c]);
    }
    }
    }
}
