// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    var i3: u32 = 0u;
    var i4: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pa: array<f32, 64>;
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    i1 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    i2 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + ((lidx + (k * 256u)) / 64u)) < cfg.m)) {
    if ((((cfg.r_off + (wid.y * 64u)) + ((lidx + (k * 256u)) % 64u)) < cfg.d)) {
    f0 = phi[((((t + ((lidx + (k * 256u)) / 64u)) * cfg.d) + (cfg.r_off + (wid.y * 64u))) + ((lidx + (k * 256u)) % 64u))];
    }
    if ((((wid.x * 64u) + ((lidx + (k * 256u)) % 64u)) < cfg.d)) {
    f1 = phi[((((t + ((lidx + (k * 256u)) / 64u)) * cfg.d) + (wid.x * 64u)) + ((lidx + (k * 256u)) % 64u))];
    }
    }
    wg_a[(((lidx + (k * 256u))) & 2047u)] = f0;
    wg_b[(((lidx + (k * 256u))) & 2047u)] = f1;
    }
    workgroupBarrier();
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[0u] = (pa[0u] + (wg_a[((((k * 64u) + lid.y)) & 2047u)] * wg_b[((((k * 64u) + lid.x)) & 2047u)]));
    pa[1u] = (pa[1u] + (wg_a[((((k * 64u) + lid.y)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    pa[2u] = (pa[2u] + (wg_a[((((k * 64u) + lid.y)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    pa[3u] = (pa[3u] + (wg_a[((((k * 64u) + lid.y)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    pa[4u] = (pa[4u] + (wg_a[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_b[((((k * 64u) + lid.x)) & 2047u)]));
    pa[5u] = (pa[5u] + (wg_a[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    pa[6u] = (pa[6u] + (wg_a[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    pa[7u] = (pa[7u] + (wg_a[(((((k * 64u) + lid.y) + 16u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    pa[8u] = (pa[8u] + (wg_a[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_b[((((k * 64u) + lid.x)) & 2047u)]));
    pa[9u] = (pa[9u] + (wg_a[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    pa[10u] = (pa[10u] + (wg_a[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    pa[11u] = (pa[11u] + (wg_a[(((((k * 64u) + lid.y) + 32u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    pa[12u] = (pa[12u] + (wg_a[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_b[((((k * 64u) + lid.x)) & 2047u)]));
    pa[13u] = (pa[13u] + (wg_a[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 16u)) & 2047u)]));
    pa[14u] = (pa[14u] + (wg_a[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 32u)) & 2047u)]));
    pa[15u] = (pa[15u] + (wg_a[(((((k * 64u) + lid.y) + 48u)) & 2047u)] * wg_b[(((((k * 64u) + lid.x) + 48u)) & 2047u)]));
    }
    workgroupBarrier();
    }
    i3 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    i4 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((((wid.y * 64u + lid.y) + min(k, 3u) * 16u) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.d)) & (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u) < cfg.d))) {
    if (((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows) & ((((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d) & ((cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d)) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
}
