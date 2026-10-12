// grid 16 16 1 1
// The u32 forms and the bit, dot and modulo overloads (2026-10-12): bit functions on an index, on vec2<u32>, insertBits, unpack, ~, integer dot, % on f32.
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
    n = n ^ 2u;
    n = (n | 1u) & 7u;
    let c = countOneBits(n) + countLeadingZeros(n) + countTrailingZeros(n) + firstLeadingBit(n) + firstTrailingBit(n);
    let cv = countOneBits(vec2<u32>(n, 3u)) + reverseBits(vec2<u32>(n, 1u)) & vec2<u32>(255u);
    let x = extractBits(n, 1u, 2u) + insertBits(n, 5u, 2u, 3u) + (~n & 15u) + (reverseBits(n) >> 24u);
    let p = pack4x8unorm(v) & 255u;
    let u = unpack4x8unorm(p).x + unpack2x16snorm(n).y + unpack2x16float(n).x * 0.0;
    g = f32(c) + f32(cv.y) + f32(x) + u + f32(dot(vec2<u32>(n, 1u), vec2<u32>(2u, 3u))) + f0 % 0.75;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
