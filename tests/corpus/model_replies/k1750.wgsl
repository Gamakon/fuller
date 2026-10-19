@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let l = wid.x * 64u + lid;
    let valid = l < n_band;
    let base = l * dp;
    let nblk = (c + 15u) / 16u;
    let nchunk = (dp + 63u) / 64u;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;
    for (var cbi = 0u; cbi < nblk; cbi += 1u) {
        let cb = cbi * 16u;
        var acc: array<f32, 16>;
        for (var k = 0u; k < 16u; k += 1u) { acc[k] = 0.0; }
        for (var chi = 0u; chi < nchunk; chi += 1u) {
            let d0 = chi * 64u;
            workgroupBarrier();
            for (var i = 0u; i < 16u; i += 1u) {
                let idx = i * 64u + lid;
                let cc = idx >> 6u;
                let j = idx & 63u;
                let ci = cb + cc;
                let d = d0 + j;
                var v: f32 = 0.0;
                if (ci < c && d < dp) { v = centroid_keys[ci * dp + d]; }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                for (var j = 0u; j < 64u; j += 1u) {
                    let d = d0 + j;
                    var x: f32 = 0.0;
                    if (d < dp) { x = point_keys[base + d]; }
                    for (var k = 0u; k < 16u; k += 1u) {
                        acc[k] = acc[k] + x * tile[k * 64u + j];
                    }
                }
            }
        }
        for (var k = 0u; k < 16u; k += 1u) {
            let ci = cb + k;
            if (ci < c && acc[k] > best_sim) { best_sim = acc[k]; best = ci; }
        }
    }
    if (valid) { assignments[row_off + l] = best; }
}
