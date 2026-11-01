// ROW MATVEC READ BASELINE, bandwidth-shaped: one 32-wide workgroup per matrix row; each invocation takes whole blocks
// (jb = lane, lane + 32, ...), loads its 8 nibble words as two vec4<u32> and its (scale, bias), and does the block's
// 64-term dot product in registers; one subgroupAdd per row.
struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32, bpw: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> words: array<vec4<u32>>;
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
    for (var jb = lane; jb < rs.y; jb += 32u) {
        let blk = rs.x + jb;
        let p = params[blk];
        let w = array<vec4<u32>, 2>(words[blk * 2u], words[blk * 2u + 1u]);
        var d = 0.0; var sx = 0.0;
        for (var c = 0u; c < 64u; c++) {
            let word = w[c / 32u][(c / 8u) % 4u];
            let x = 0.5 + 0.01 * f32(jb * 64u + c);
            d += f32((word >> ((c % 8u) * 4u)) & 15u) * x;
            sx += x;
        }
        acc += d * p.z + p.w * sx;
    }
    var dot = subgroupAdd(acc);
    if (ssize != 32u) { dot = bitcast<f32>(0x7fc00000u); }
    if (lane == 0u) { out[row] = dot; }
}
