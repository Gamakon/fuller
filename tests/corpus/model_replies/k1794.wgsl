@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

const CT: u32 = 8u;
const DT: u32 = 256u;
const WG: u32 = 128u;

var<workgroup> tile: array<f32, 2048>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let local = gid.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let valid = local < n_band;
    let pbase = local * dp;
    var best: u32 = 0u;
    var best_sim: f32 = -3.4e38;

    for (var cb: u32 = 0u; cb < c; cb += CT) {
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var a4: f32 = 0.0;
        var a5: f32 = 0.0;
        var a6: f32 = 0.0;
        var a7: f32 = 0.0;
        for (var db: u32 = 0u; db < dp; db += DT) {
            workgroupBarrier();
            for (var i: u32 = 0u; i < CT * DT; i += WG) {
                let idx = i + lid;
                let ci = idx / DT;
                let d = idx % DT;
                let cg = cb + ci;
                let dd = db + d;
                var v: f32 = 0.0;
                if (cg < c && dd < dp) {
                    v = centroid_keys[cg * dp + dd];
                }
                tile[idx] = v;
            }
            workgroupBarrier();
            if (valid) {
                let rem = min(DT, dp - db);
                for (var d: u32 = 0u; d < DT; d += 1u) {
                    if (d < rem) {
                        let pv = point_keys[pbase + db + d];
                        a0 += pv * tile[d];
                        a1 += pv * tile[DT + d];
                        a2 += pv * tile[2u * DT + d];
                        a3 += pv * tile[3u * DT + d];
                        a4 += pv * tile[4u * DT + d];
                        a5 += pv * tile[5u * DT + d];
                        a6 += pv * tile[6u * DT + d];
                        a7 += pv * tile[7u * DT + d];
                    }
                }
            }
        }
        if (cb < c && a0 > best_sim) { best_sim = a0; best = cb; }
        if (cb + 1u < c && a1 > best_sim) { best_sim = a1; best = cb + 1u; }
        if (cb + 2u < c && a2 > best_sim) { best_sim = a2; best = cb + 2u; }
        if (cb + 3u < c && a3 > best_sim) { best_sim = a3; best = cb + 3u; }
        if (cb + 4u < c && a4 > best_sim) { best_sim = a4; best = cb + 4u; }
        if (cb + 5u < c && a5 > best_sim) { best_sim = a5; best = cb + 5u; }
        if (cb + 6u < c && a6 > best_sim) { best_sim = a6; best = cb + 6u; }
        if (cb + 7u < c && a7 > best_sim) { best_sim = a7; best = cb + 7u; }
    }
    if (valid) {
        assignments[row_off + local] = best;
    }
}
