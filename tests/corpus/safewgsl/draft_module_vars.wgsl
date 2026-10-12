// grid 16 16 1 1
// Module scope (2026-10-12): a const, an override at its default, var<private> with and without an initializer, a workgroup scalar read by workgroupUniformLoad.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
const SCALE: f32 = 0.5;
override BIAS: f32 = 0.25;
var<private> pg: f32 = 1.5;
var<private> pa: array<f32, 8>;
var<workgroup> wv: f32;
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
    pg = pg + SCALE * f0;
    pa[n & 7u] = f0 + BIAS;
    if (lidx == 0u) { wv = f0; }
    let u = workgroupUniformLoad(&wv);
    g = pg + pa[n & 7u] + u;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
