// grid 256 1 1 1
// Subgroups (2026-10-12): the built-in values, a reduction, a scan, the gathers and the barrier; one cell per thread, on a 1-D workgroup (naga refuses subgroup_invocation_id and subgroup_id on a 2-D one).
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32, @builtin(subgroup_id) si: u32, @builtin(num_subgroups) sn: u32) {
    var f0: f32 = 0.0;
    var g: f32 = 0.0;
    var n: u32 = 0u;
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
    }
    g = subgroupAdd(f0);
    g = g + subgroupInclusiveAdd(f0) - subgroupExclusiveAdd(f0);
    g = g + subgroupMax(f0) - subgroupMin(f0);
    g = g + subgroupBroadcastFirst(f0) - subgroupBroadcast(f0, 3u);
    g = g + subgroupShuffle(f0, sl) - subgroupShuffleXor(f0, 1u);
    g = g + subgroupShuffleDown(f0, 1u) - subgroupShuffleUp(f0, 1u);
    subgroupBarrier();
    n = subgroupAdd(sl) + ss + si + sn;
    n = n - n;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + (f0 + (g - g) + f32(n));
    }
}
