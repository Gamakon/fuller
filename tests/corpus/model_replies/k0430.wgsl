// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>;
    for (k = 0u; k < 32u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    pb[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < (((cfg.m + 15u) / 16u)); t = t + 1u) {
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if ((((t * 16u) + ((lidx + (k * 128u)) / 64u)) < cfg.m)) {
    if (((((wid.y * 64u) + ((lidx + (k * 128u)) % 64u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + ((lidx + (k * 128u)) % 64u))) < cfg.d))) {
    f0 = phi[(((((t * 16u) + ((lidx + (k * 128u)) / 64u)) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((lidx + (k * 128u)) % 64u)))];
    }
    if ((((wid.x * 64u) + ((lidx + (k * 128u)) % 64u)) < cfg.d)) {
    f1 = phi[((((t * 16u) + ((lidx + (k * 128u)) / 64u)) * cfg.d) + ((wid.x * 64u) + ((lidx + (k * 128u)) % 64u)))];
    }
    }
    wg_a[(((lidx + (k * 128u))) & 2047u)] = f0;
    wg_b[(((lidx + (k * 128u))) & 2047u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 2u) {
    for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    pa[((((k3 * 8u) + k2)) & 63u)] = (pa[((((k3 * 8u) + k2)) & 63u)] + (wg_a[(((((k * 64u) + lid.y) + (8u * k2))) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + (16u * k3))) & 2047u)]));
    pb[((((k3 * 8u) + k2)) & 63u)] = (pb[((((k3 * 8u) + k2)) & 63u)] + (wg_a[((((((k * 64u) + 64u) + lid.y) + (8u * k2))) & 2047u)] * wg_b[((((((k * 64u) + 64u) + lid.x) + (16u * k3))) & 2047u)]));
    }
    }
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((wid.y * 64u + lid.y) + k * 8u) < cfg.r_rows) && (((wid.x * 64u + lid.x) + k2 * 16u) < cfg.d) && ((cfg.r_off + ((wid.y * 64u + lid.y) + k * 8u)) < cfg.d)) { h[((((wid.y * 64u + lid.y) + k * 8u)) * cfg.d + (((wid.x * 64u + lid.x) + k2 * 16u)))] = h[((((wid.y * 64u + lid.y) + k * 8u)) * cfg.d + (((wid.x * 64u + lid.x) + k2 * 16u)))] + (pa[((((k2 * 8u) + k)) & 63u)] + pb[((((k2 * 8u) + k)) & 63u)]); }
    }
    }
}
