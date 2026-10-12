// grid 16 16 1 1
// interface cosine
// A dense batched cosine re-score (UltraDim's exact re-score prototype, 2026-10-12): score[q][c] = <qry[q], cand[c]> / (|qry[q]| |cand[c]|), the dot product summed with Kahan's compensation; a thread owns one (query, candidate) cell and adds its score to it (the host zeroes score, as the Gram's host zeroes h).
struct Params { nc: u32, dim: u32, nq: u32, q_off: u32, };
@group(0) @binding(0) var<storage, read> cand: array<f32>;
@group(0) @binding(1) var<storage, read_write> score: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> qry: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var f2: f32 = 0.0;
    var f3: f32 = 0.0;
    let qr = p.q_off + gid.y;
    for (var t: u32 = 0u; t < p.dim; t = t + 1u) {
        let a = qry[qr * p.dim + t];
        let b = cand[gid.x * p.dim + t];
        let y = a * b - f1;
        let s = f0 + y;
        f1 = (s - f0) - y;
        f0 = s;
        f2 = f2 + a * a;
        f3 = f3 + b * b;
    }
    if (gid.y < p.nq && gid.x < p.nc) {
        score[gid.y * p.nc + gid.x] = score[gid.y * p.nc + gid.x] + f0 / sqrt(max(f2 * f3, 1e-30));
    }
}
