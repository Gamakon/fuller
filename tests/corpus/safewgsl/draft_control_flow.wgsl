// grid 16 16 1 1
// Control flow (2026-10-12): while, a for that breaks, a for from 2 that continues, a do-while (break if), a switch with a case list and default.
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
    var w: u32 = 0u;
    while (w < 4u) { g = g + 1.0; w = w + 1u; }
    for (var j: u32 = 0u; j < 8u; j = j + 1u) { if (f32(j) > f0) { break; } g = g + 1.0; }
    for (var j2: u32 = 2u; j2 < 8u; j2 = j2 + 1u) { if ((j2 & 1u) == 1u) { continue; } g = g + f32(j2); }
    var w2: u32 = 0u;
    loop { g = g + 0.5; continuing { w2 = w2 + 1u; break if w2 >= 3u; } }
    switch (n & 3u) { case 0u: { g = g + 1.0; } case 1u, 2u: { g = g + 2.0; } default: { g = g + 3.0; } }
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
