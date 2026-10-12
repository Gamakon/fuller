// grid 16 16 1 1
// Vectors (2026-10-12): the thread id as an operand, select on vec2<bool>, bitcast of a vec4, subgroupBallot into a vec4<u32> register, a private array of vec2.
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
    let q = gid + vec3<u32>(1u);
    let sel = select(vec2<f32>(1.0), vec2<f32>(2.0), vec2<bool>(f0 > 1.0, true));
    let bits = bitcast<vec4<f32>>(bitcast<vec4<u32>>(v));
    let ballot = subgroupBallot(f0 > 0.5);
    var a: array<vec2<f32>, 4>;
    a[n & 3u] = vec2<f32>(f0, 1.0);
    g = f32(q.x * q.z & 7u) + sel.y + bits.w + f32(countOneBits(ballot.x)) + a[n & 3u].y;
    if (gid.y < cfg.r_rows && gid.x < cfg.d && gr < cfg.d) {
        h[gid.y * cfg.d + gid.x] = h[gid.y * cfg.d + gid.x] + f0 + g * 0.0;
    }
}
