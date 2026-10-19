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

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(256, 1, 1)
fn generated_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = (gid.z * nwg.x * 256u * nwg.y) + (gid.y * nwg.x * 256u) + gid.x;
    let width = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let off = buffer_6_global[0u];
    let noise_lo = buffer_6_global[1u];
    let noise_tail = buffer_6_global[2u] ^ buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let omleak = 1.0 - leak;

    let base = id * 4u + off;
    if base >= width {
        return;
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let r = base + j;
        if r < width {
            let hr = fmix(key_lo ^ r);
            let rb = r * 10u;
            var sum = 0.0;
            var c = 0.0;
            var ax = 0.0;
            for (var s = 0u; s < 10u; s = s + 1u) {
                let src = buffer_0_global[rb + s];
                let xv = buffer_1_global[src];
                let hh = fmix(hr ^ (key_hi ^ s));
                let w = buffer_2_global[hh >> 22u];
                let p = w * xv;
                let y = p - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
                ax = ax + xv;
            }
            let hn = fmix(fmix(noise_lo ^ r) ^ noise_tail);
            let z = buffer_2_global[hn >> 22u];
            let pre = scale * sum + loc * ax + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = omleak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
