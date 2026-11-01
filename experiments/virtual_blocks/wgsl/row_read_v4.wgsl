// ROW MATVEC TIMING BASELINE (32 invocations, 2 cells each, as row_gen_v4.wgsl): one workgroup per matrix row of layer 13's MLP; each lane reads its cell of every block
// of the row (8 packed u32 a block), dequantises, multiplies the activations (the same synthetic x as row_gen.wgsl)
// and accumulates; ONE reduction per row.
struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32, bpw: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tabs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Meta;
fn row_span(r: u32) -> vec2<u32> {
    if (r < 4864u) { return vec2<u32>(r * 14u, 14u); }
    if (r < 9728u) { return vec2<u32>(68096u + (r - 4864u) * 14u, 14u); }
    return vec2<u32>(136192u + (r - 9728u) * 76u, 76u);
}
@compute @workgroup_size(32)
fn gen_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(subgroup_size) ssize: u32) {
    let row = wid.y * cfg.groups_x + wid.x;
    let rs = row_span(row);
    var acc = 0.0;
    for (var jb = 0u; jb < rs.y; jb++) {
        let blk = rs.x + jb;
        let p = params[blk];
        let w0 = tabs[blk * 8u + lane / 8u];
        let w1 = tabs[blk * 8u + 4u + lane / 8u];
        let sh = (lane % 8u) * 4u;
        acc += (f32((w0 >> sh) & 15u) * p.z + p.w) * (0.5 + 0.01 * f32(jb * 64u + lane))
             + (f32((w1 >> sh) & 15u) * p.z + p.w) * (0.5 + 0.01 * f32(jb * 64u + lane + 32u));
    }
    var dot = subgroupAdd(acc);
    if (ssize != 32u) { dot = bitcast<f32>(0x7fc00000u); }
    if (lane == 0u) { out[row] = dot; }
}
