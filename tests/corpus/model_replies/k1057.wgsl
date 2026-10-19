@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

fn fmix(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // each thread handles 2 rows (adjacent across the workgroup); the upper half of the grid exits
    if (wid.x * 2u >= nwg.x) {
        return;
    }
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let n_lo = buffer_6_global[1u];
    let n_hi = buffer_6_global[2u];
    let tstep = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let nkey = n_hi ^ tstep;
    let base = off + wid.x * 512u + lid.x;

    for (var j = 0u; j < 2u; j += 1u) {
        let r = base + j * 256u;
        if (r < width) {
            let h1 = fmix(key_lo ^ r);
            let rb = r * 10u;
            var s = 0.0;
            var c = 0.0;
            var ax = 0.0;
            for (var k = 0u; k < 10u; k += 1u) {
                let src = buffer_0_global[rb + k];
                let x = buffer_1_global[src];
                let hh = fmix(h1 ^ (key_hi ^ k));
                let cb = buffer_2_global[hh >> 22u];
                let v = cb * x - c;
                let t = s + v;
                c = (t - s) - v;
                s = t;
                ax = ax + x;
            }
            let z = buffer_2_global[fmix(fmix(n_lo ^ r) ^ nkey) >> 22u];
            let pre = scale * s + loc * ax + buffer_3_global[r] + sigma * z;
            var th = 1.0;
            if (pre <= 43.0) {
                th = tanh(pre);
            }
            buffer_4_global[r] = (1.0 - leak) * buffer_1_global[r] + leak * th;
        }
    }
}
