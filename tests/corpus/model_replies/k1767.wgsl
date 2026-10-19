@group(0) @binding(0) var<storage, read> point_keys: array<f32>;
@group(0) @binding(1) var<storage, read> centroid_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    let n_band = params.x;
    let c = params.y;
    let dp = params.z;
    let row_off = params.w;
    let base = wid.x * 256u;

    var off: vec4<u32>;
    for (var t = 0u; t < 4u; t += 1u) {
        let l = base + lid + 64u * t;
        let lc = min(l, n_band - 1u);
        off[t] = lc * dp;
    }

    var bs = vec4<f32>(-3.4e38);
    var bi = vec4<u32>(0u);

    for (var cb = 0u; cb < c; cb += 8u) {
        var acc: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j += 1u) {
            acc[j] = vec4<f32>(0.0);
        }
        for (var d0 = 0u; d0 < dp; d0 += 128u) {
            workgroupBarrier();
            for (var i = 0u; i < 1024u; i += 64u) {
                let j = i + lid;
                let r = j >> 7u;
                let k = j & 127u;
                let ci = cb + r;
                let d = d0 + k;
                var v = 0.0;
                if (ci < c && d < dp) {
                    v = centroid_keys[ci * dp + d];
                }
                tile[j] = v;
            }
            workgroupBarrier();
            for (var k = 0u; k < 128u; k += 1u) {
                let d = d0 + k;
                if (d < dp) {
                    let pv = vec4<f32>(
                        point_keys[off.x + d],
                        point_keys[off.y + d],
                        point_keys[off.z + d],
                        point_keys[off.w + d]);
                    for (var j = 0u; j < 8u; j += 1u) {
                        acc[j] += pv * tile[j * 128u + k];
                    }
                }
            }
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let ci = cb + j;
            if (ci < c) {
                for (var t = 0u; t < 4u; t += 1u) {
                    if (acc[j][t] > bs[t]) {
                        bs[t] = acc[j][t];
                        bi[t] = ci;
                    }
                }
            }
        }
    }

    for (var t = 0u; t < 4u; t += 1u) {
        let l = base + lid + 64u * t;
        if (l < n_band) {
            assignments[row_off + l] = bi[t];
        }
    }
}
