// FUSED TIMING BASELINE: read a stored 4-bit block (8 packed u32), dequantise, multiply the activations (the same
// synthetic x as fused_gen.wgsl), sum with the same subgroup reduction, write one f32 per block.
struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32, bpw: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tabs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Meta;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn gen_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(subgroup_invocation_id) sid: u32, @builtin(subgroup_size) ssize: u32) {
    for (var jb = 0u; jb < cfg.bpw; jb++) {
    let blk = (wid.y * cfg.groups_x + wid.x) * cfg.bpw + jb;
    let valid = blk < cfg.n_blocks;
    let b = min(blk, cfg.n_blocks - 1u);
    let p = params[b];
    let w = tabs[b * 8u + lane / 8u];
    let y = f32((w >> ((lane % 8u) * 4u)) & 15u) * p.z + p.w;
    let xv = 0.5 + 0.01 * f32((blk % 14u) * 64u + lane);
    let part = subgroupAdd(y * xv);
    if (sid == 0u) { red[lane / ssize] = part; }
    workgroupBarrier();
    var dot = 0.0;
    for (var g = 0u; g < 64u / ssize; g++) { dot += red[g]; }
    if (valid && lane == 0u) { out[blk] = dot; }
    workgroupBarrier();
    }
}
