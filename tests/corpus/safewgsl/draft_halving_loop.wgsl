// grid 256 1 1 1
// A tree reduction written as a halving loop (2026-10-12): each thread's dot product into workgroup memory, then `for (var s = 128u; s > 0u; s = s >> 1u)` with the step under `lidx < s` and a barrier, the workgroup's sum added to its row's first cell.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    let gr = cfg.r_off + gid.y;
    for (var t: u32 = 0u; t < cfg.m; t = t + 1u) {
        f0 = f0 + phi[t * cfg.d + gr] * phi[t * cfg.d + gid.x];
    }
    part[lidx] = f0;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (lidx < s) {
            part[lidx] = part[lidx] + part[lidx + s];
        }
        workgroupBarrier();
    }
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + (f0 + (part[0] - part[0]));
    }
}
