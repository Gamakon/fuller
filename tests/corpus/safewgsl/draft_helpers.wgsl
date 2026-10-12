// grid 16 16 1 1
// Helpers (2026-10-12): a pointer parameter, guards before the last return, an if-else of two returns.
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
fn bump(p: ptr<function, f32>, by: f32) { *p = *p + by; }
fn clip(x: f32) -> f32 { if (x > 1.0) { return 1.0; } if (x < -1.0) { return -1.0; } return x; }
fn pick(x: f32) -> f32 { if (x > 0.5) { return x; } else { return 0.5; } }
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
    bump(&g, f0);
    g = g + clip(f0) + pick(f0);
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
