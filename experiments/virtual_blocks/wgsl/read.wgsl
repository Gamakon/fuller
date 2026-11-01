// Baseline: READ a stored 4-bit block (8 packed u32 = 64 nibbles) and dequantise it, q*scale + bias.
// Same bindings and dispatch as vblock.wgsl, so the two are timed the same way.
struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32 }
@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tabs: array<u32>;          // n_blocks * 8 packed nibble words
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Meta;
@compute @workgroup_size(64)
fn gen_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let blk = wid.y * cfg.groups_x + wid.x;
    if (blk >= cfg.n_blocks) { return; }
    let p = params[blk];
    let w = tabs[blk * 8u + lane / 8u];
    let q = f32((w >> ((lane % 8u) * 4u)) & 15u);
    out[blk * 64u + lane] = q * p.z + p.w;
}
