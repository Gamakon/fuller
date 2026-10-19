@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l = gid.x;
    let valid = l < n_band;
    let pbase = l * dp;
    let t = lid.x;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    let nblk = (c + 7u) / 8u;
    let nchunk = (dp + 127u) / 128u;
    for (var cb = 0u; cb < nblk; cb += 1u) {
        let c0 = cb * 8u;
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var a4: f32 = 0.0;
        var a5: f32 = 0.0;
        var a6: f32 = 0.0;
        var a7: f32 = 0.0;
        for (var ch = 0u; ch < nchunk; ch += 1u) {
            let d0 = ch * 128u;
            workgroupBarrier();
            for (var r = 0u; r < 4u; r += 1u) {
                let idx = r * 256u + t;
                let cc = idx / 128u;
                let dd = idx % 128u;
                var v: f32 = 0.0;
                if (c0 + cc < c && d0 + dd < dp) {
                    v = centroid_keys[(c0 + cc) * dp + d0 + dd];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var k = 0u; k < 128u; k += 1u) {
                    if (d0 + k < dp) {
                        let p = point_keys[pbase + d0 + k];
                        a0 = a0 + p * tile[k];
                        a1 = a1 + p * tile[128u + k];
                        a2 = a2 + p * tile[256u + k];
                        a3 = a3 + p * tile[384u + k];
                        a4 = a4 + p * tile[512u + k];
                        a5 = a5 + p * tile[640u + k];
                        a6 = a6 + p * tile[768u + k];
                        a7 = a7 + p * tile[896u + k];
                    }
                }
            }
        }
        if (c0 < c && a0 > best_sim) { best_sim = a0; best = c0; }
        if (c0 + 1u < c && a1 > best_sim) { best_sim = a1; best = c0 + 1u; }
        if (c0 + 2u < c && a2 > best_sim) { best_sim = a2; best = c0 + 2u; }
        if (c0 + 3u < c && a3 > best_sim) { best_sim = a3; best = c0 + 3u; }
        if (c0 + 4u < c && a4 > best_sim) { best_sim = a4; best = c0 + 4u; }
        if (c0 + 5u < c && a5 > best_sim) { best_sim = a5; best = c0 + 5u; }
        if (c0 + 6u < c && a6 > best_sim) { best_sim = a6; best = c0 + 6u; }
        if (c0 + 7u < c && a7 > best_sim) { best_sim = a7; best = c0 + 7u; }
    }
    if (valid) {
        assignments[row_off + l] = best;
    }
}
