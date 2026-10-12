// grid 16 16 1 1
// Abstract float literals beside an f32 in built-in calls (2026-10-12): naga 0.20 lowers each as As { convert: Some(4) } f32 to f32.
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
    g = max(f0, 1.0) + min(f0, 1.0) + clamp(f0, 0.0, 1.0) + mix(f0, 1.0, 0.25) + pow(abs(f0), 2.0) + step(0.5, f0) + smoothstep(0.0, 1.0, f0);
    g = g + fma(f0, 2.0, 1.0) + atan2(f0, 2.0) + distance(f0, 1.0) + ldexp(f0, 2) + dot(v, v) + length(v) + normalize(v).x + reflect(v, v).x + faceForward(v, v, v).x;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
